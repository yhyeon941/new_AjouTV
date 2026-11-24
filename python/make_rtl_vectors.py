#!/usr/bin/env python
import argparse
import os
import shutil
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from q88_golden import (THRESHOLD_Q88, Roms, attn_output, export_mem, fc_q88, load_mem, make_roms, qkv, scores,
                        softmax_q88, to_q88, wrap16)


def hexw(v):
    return f"{int(v) & 0xFFFF:04X}"


def write_words(path, arr):
    with open(path, "w") as f:
        for v in np.asarray(arr).reshape(-1):
            f.write(hexw(v) + "\n")


def edge_windows(rs, n_random):
    ws = [np.zeros((4, 4)), np.full((4, 4), 0x7FFF), np.full((4, 4), -0x8000),
          np.full((4, 4), 256), np.full((4, 4), -256)]
    alt = np.zeros((4, 4))
    alt[::2, :] = 0x0200
    alt[1::2, :] = -0x0200
    ws.append(alt)
    for _ in range(n_random):                       # 일반 범위 (표준화 입력 +-3 정도)
        ws.append(rs.integers(-768, 768, size=(4, 4)))
    for _ in range(n_random // 2):                  # 넓은 범위 (wrap/포화 경로까지 비트 단위로 확인)
        ws.append(rs.integers(-0x4000, 0x4000, size=(4, 4)))
    return np.array(ws, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "orig"], required=True)
    ap.add_argument("--weights", default=None, help="train_attention.py 가 만든 model_fp32.npz (fixed 모드)")
    ap.add_argument("--demo", action="store_true", help="무작위 가중치로 키트를 만든다 (배선 점검용)")
    ap.add_argument("--arch", default="res_square", choices=["res_square", "square", "rtl"], help="--demo 일 때 구조")
    ap.add_argument("--data", default=None, help="dataset 폴더 (Test 윈도우를 입력으로 사용)")
    ap.add_argument("--mem-dir", required=True, help="저장소 mem 폴더 (softmax ROM 및 orig 모드 가중치)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-data", type=int, default=120, help="데이터셋에서 가져올 윈도우 수")
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rs = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "orig":
        m = wrap16(load_mem(os.path.join(args.mem_dir, "weights.mem")))
        f = wrap16(load_mem(os.path.join(args.mem_dir, "fc_weights.mem")))
        wi = dict(wq=m[:16].reshape(4, 4), wk=m[16:32].reshape(4, 4), wv=m[32:].reshape(4, 4), wfc=f)
        arch, b_int, c_int = "rtl", 0, None
        for nm in ("weights.mem", "fc_weights.mem"):
            shutil.copy(os.path.join(args.mem_dir, nm), os.path.join(args.out, nm))
        kw = dict(mul_bits=16, fc_signed=False, slice_start=4, inv_mode="rtl")
    else:
        if args.demo:
            w = dict(wq=rs.normal(0, 0.35, (4, 4)), wk=rs.normal(0, 0.35, (4, 4)), wv=rs.normal(0, 0.8, (4, 4)),
                     wfc=rs.normal(0, 0.25, 16), arch=np.array(args.arch))
            if args.arch != "rtl":
                w["b"] = np.array(-0.3)
            print("[demo] 무작위 가중치로 키트를 만듭니다. 실제 모델 값이 아닙니다.")
        else:
            if not args.weights:
                sys.exit("--weights model_fp32.npz 또는 --demo 가 필요합니다.")
            z = np.load(args.weights)
            w = {k: z[k].astype(np.float64) for k in ("wq", "wk", "wv", "wfc")}
            w["arch"] = np.array(str(z["arch"]) if "arch" in z.files else "rtl")
            for k in ("b", "c"):
                if k in z.files:
                    w[k] = z[k].astype(np.float64)
        arch = str(w["arch"])
        if arch not in ("res_square", "square", "rtl"):
            sys.exit(f"이 키트의 RTL 은 res_square 구조용입니다 (가중치 arch={arch}).")
        export_mem(args.out, w)
        wi = {k: to_q88(v) for k, v in w.items() if k in ("wq", "wk", "wv", "wfc", "b", "c")}
        b_int, c_int = (int(wi["b"]) if "b" in wi else 0), wi.get("c")
        kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full")
        print(f"구조 {arch}: weights.mem, fc_weights.mem" + (", fc_extra.mem" if arch != "rtl" else "") + " 생성")
    for nm in ("softmax_exp_rom.mem", "softmax_inv_rom.mem"):
        shutil.copy(os.path.join(args.mem_dir, nm), os.path.join(args.out, nm))
    roms = make_roms("repo", args.mem_dir)

    parts = []
    if args.data:
        te = np.load(os.path.join(args.data, "dataset_test.npz"))
        idx = rs.choice(len(te["y"]), min(args.n_data, len(te["y"])), replace=False)
        parts.append(te["X_q88"][idx].astype(np.int64))
    parts.append(edge_windows(rs, args.n_random))
    X = np.concatenate(parts)
    X = wrap16(X)

    stats = {}
    q, k, v = qkv(X, wi["wq"], wi["wk"], wi["wv"], stats)
    sc = scores(q, k, stats)
    aw = softmax_q88(sc, roms, kw["mul_bits"], kw["slice_start"], kw["inv_mode"])
    ao = attn_output(aw, v, stats)
    z = fc_q88(ao, wi["wfc"], kw["fc_signed"], stats, arch=arch, b_int=b_int, c_int=c_int, xq=X)
    label = (z > THRESHOLD_Q88).astype(np.int64)
    colsum = aw.sum(axis=1)
    cause = colsum.argmax(axis=1)

    n = len(X)
    write_words(os.path.join(args.out, "vec_n.mem"), [n])
    write_words(os.path.join(args.out, "vec_x.mem"), X)
    for nm, arr in (("q", q), ("k", k), ("v", v), ("score", sc), ("aw", aw), ("ao", ao)):
        write_words(os.path.join(args.out, f"exp_{nm}.mem"), arr)
    write_words(os.path.join(args.out, "exp_z.mem"), z)
    write_words(os.path.join(args.out, "exp_label.mem"), label)
    write_words(os.path.join(args.out, "exp_cause.mem"), cause)
    print(f"윈도우 {n}개, 이상 판정 {int(label.sum())}개, 중간 wrap 발생 비율 { {k: round(v, 4) for k, v in stats.items()} }")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()