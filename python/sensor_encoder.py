#!/usr/bin/env python
import argparse
import json

import numpy as np

SENSORS = ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE", "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"]
SCALE = 16
CLIP_Z = 4.0
SLEW_STEP = 0.5


def standardize(raw, mean, std, clip=CLIP_Z):
    """raw: (4,) 또는 (4,T) 센서 x 시점. mean/std: 길이 4."""
    raw = np.asarray(raw, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64).reshape(4, *([1] * (raw.ndim - 1)))
    s = np.asarray(std, dtype=np.float64).reshape(4, *([1] * (raw.ndim - 1)))
    return np.clip((raw - m) / s, -clip, clip)


def slew_limit(z, max_step=SLEW_STEP):
    z = np.asarray(z, dtype=np.float64)
    out = z.copy()
    for t in range(1, z.shape[1]):
        out[:, t] = out[:, t - 1] + np.clip(z[:, t] - out[:, t - 1], -max_step, max_step)
    return out


def prepare_stream(raw, mean, std, clip=CLIP_Z, slew=SLEW_STEP):
    z = standardize(raw, mean, std, clip)
    return z if slew is None else slew_limit(z, slew)


def to_q44(z):
    return np.clip(np.rint(np.asarray(z, dtype=np.float64) * SCALE), -128, 127).astype(np.int8)


def stream_to_bytes(z):
    q = to_q44(z)
    return bytes(int(q[s, t]) & 0xFF for t in range(q.shape[1]) for s in range(4))


def byte_to_q88(b):
    s = b - 256 if b >= 128 else b
    return s * 16


def send_stream(port, baud, z, delay_s=0.0):
    import time

    import serial  # pip install pyserial
    data = stream_to_bytes(z)
    with serial.Serial(port, baud, timeout=1) as ser:
        for i in range(0, len(data), 4):
            ser.write(data[i:i + 4])
            if delay_s:
                time.sleep(delay_s)


class SlidingWindowModel:

    def __init__(self):
        self.buf = [0] * 4
        self.cnt = 0
        self.filled = 0
        self.window = [[0] * 4 for _ in range(4)]            # window[센서][시간]
        self.ready = False

    def push(self, q88):
        self.ready = False
        self.buf[self.cnt] = q88
        if self.cnt == 3:
            for i in range(4):
                self.window[i] = self.window[i][1:] + [self.buf[i] if i < 3 else q88]
            self.cnt = 0
            self.filled = min(self.filled + 1, 4)
            if self.filled >= 4:
                self.ready = True
        else:
            self.cnt += 1


def selftest():
    rs = np.random.default_rng(0)
    z = rs.uniform(-8, 8, size=(4, 50))
    data = stream_to_bytes(z)
    q88 = np.array([byte_to_q88(b) for b in data]).reshape(50, 4).T
    expect = np.rint(z * SCALE).clip(-128, 127) * (256 // SCALE)
    assert np.array_equal(q88, expect.astype(np.int64)), "라운드트립 불일치"
    assert to_q44(8.0) == 127 and to_q44(-8.0) == -128 and to_q44(-9.0) == -128
    assert to_q44(0.03) == 0 and to_q44(0.04) == 1 and to_q44(-0.04) == -1          # 1/32 = 0.03125 경계
    assert byte_to_q88(0x7F) == 0x07F0 and (byte_to_q88(0x80) & 0xFFFF) == 0xF800 and (byte_to_q88(0xFF) & 0xFFFF) == 0xFFF0
    zw = rs.uniform(-3, 3, size=(4, 4))
    sw = SlidingWindowModel()
    for b in stream_to_bytes(zw):
        sw.push(byte_to_q88(b))
    assert sw.ready, "16바이트 뒤에 window_ready 가 나와야 함"
    assert np.array_equal(np.array(sw.window), (np.rint(zw * SCALE).clip(-128, 127) * 16).astype(np.int64)), "센서/시간 순서 불일치"
    z5 = rs.uniform(-3, 3, size=(4, 5))
    sw = SlidingWindowModel()
    flags = []
    for b in stream_to_bytes(z5):
        sw.push(byte_to_q88(b))
        flags.append(sw.ready)
    assert flags.count(True) == 2 and np.array_equal(np.array(sw.window)[:, :], (np.rint(z5[:, 1:] * SCALE) * 16).astype(np.int64))
    zz = np.zeros((4, 12))
    zz[:, 4:] = 4.0
    lim = slew_limit(zz, 0.5)
    assert np.allclose(lim[:, 0], 0.0) and np.max(np.abs(np.diff(lim, axis=1))) <= 0.5 + 1e-12
    assert np.isclose(lim[0, 4], 0.5) and np.isclose(lim[0, 10], 3.5) and np.isclose(lim[0, 11], 4.0)     # 점프 후 8번째 시점에 4σ 도달
    raw = np.full((4, 20), 10.0)
    ps = prepare_stream(raw, np.zeros(4), np.ones(4), clip=4.0, slew=0.5)
    assert ps.max() <= 4.0 + 1e-12 and np.isclose(ps[0, 0], 4.0)
    print("자체 점검 통과: 라운드트립, 포화/반올림 경계, 바이트 순서(센서 x 시점), 시점 이동, 기울기 제한, 클리핑")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--meta", default=None, help="dataset_meta.json (mean/std)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.demo:
        if not a.meta:
            raise SystemExit("--meta dataset_t/dataset_meta.json 이 필요합니다.")
        meta = json.load(open(a.meta, encoding="utf-8"))
        mean, std = np.array(meta["mean"]), np.array(meta["std"])
        print("센서 순서:", meta.get("sensors", SENSORS))
        print("표준화 상수: mean", np.round(mean, 4).tolist(), "std", np.round(std, 4).tolist())
        for k in (-1.0, 0.0, 0.5, 1.0, 2.0):
            raw = mean + k * std
            z = standardize(raw, mean, std)
            b = stream_to_bytes(z.reshape(4, 1))
            print(f"  각 센서를 평균 + {k:+.1f} 표준편차로 보내면 바이트 = {[hex(x) for x in b]}  (Q8.8 = {[hex(byte_to_q88(x) & 0xFFFF) for x in b[:1]]} ...)")
        return
    ap.print_help()


if __name__ == "__main__":
    main()