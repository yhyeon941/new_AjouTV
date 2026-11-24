#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path


def readmemh(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"벡터 파일이 없습니다: {path}")

    data: list[int] = []
    content = re.sub(r"/\*.*?\*/", " ", path.read_text(encoding="utf-8-sig"), flags=re.S)
    index = 0
    for line in content.splitlines():
        line = line.split("//", 1)[0].split("#", 1)[0]
        for token in line.replace(",", " ").split():
            token = token.replace("_", "")
            if token.startswith("@"):
                index = int(token[1:], 16)
                continue
            if index > len(data):
                raise ValueError(f"{path.name}: @주소에 빈 구간이 있습니다 (index={index}).")
            value = int(token, 16)
            if index == len(data):
                data.append(value)
            else:
                data[index] = value
            index += 1
    return data


def load_vectors(directory: Path) -> tuple[int, int, bytes, bytes]:
    count = readmemh(directory / "e2e_n.mem")
    if len(count) < 2:
        raise ValueError("e2e_n.mem에 T와 W 두 값이 필요합니다.")
    total_steps, total_windows = count[:2]
    if total_steps < 4 or total_windows != total_steps - 3:
        raise ValueError(f"벡터 개수 오류: T={total_steps}, W={total_windows}; W=T-3이어야 합니다.")

    in_values = readmemh(directory / "uart_bytes.mem")
    out_values = readmemh(directory / "e2e_resp.mem")
    if len(in_values) < total_steps * 4:
        raise ValueError(f"uart_bytes.mem 길이 부족: {len(in_values)} < {total_steps * 4}")
    if len(out_values) < total_windows * 9:
        raise ValueError(f"e2e_resp.mem 길이 부족: {len(out_values)} < {total_windows * 9}")

    for name, values in (("uart_bytes.mem", in_values[:total_steps * 4]),
                         ("e2e_resp.mem", out_values[:total_windows * 9])):
        if any(not (0 <= x <= 255) for x in values):
            raise ValueError(f"{name}: 8비트 범위를 벗어난 값이 있습니다.")

    return (total_steps, total_windows,
            bytes(in_values[:total_steps * 4]),
            bytes(out_values[:total_windows * 9]))


def read_exact(port, byte_count: int, timeout_sec: float) -> bytes:
    deadline = time.monotonic() + timeout_sec
    received = bytearray()
    while len(received) < byte_count and time.monotonic() < deadline:
        chunk = port.read(byte_count - len(received))
        if chunk:
            received.extend(chunk)
    return bytes(received)


def hex_bytes(data: bytes) -> str:
    return data.hex(" ").upper()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Basys3 COM 포트로 4바이트 센서 데이터를 보내고 9바이트 UART 응답을 Golden Model과 비교합니다."
    )
    parser.add_argument("--port", default="COM6", help="UART 포트 (기본: COM6)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud Rate (기본: 115200)")
    parser.add_argument(
        "--vectors", type=Path, default=Path(__file__).resolve().parent / "kit_e2e",
        help="e2e_n.mem, uart_bytes.mem, e2e_resp.mem이 들어있는 폴더",
    )
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="각 UART 응답 9바이트 수신 타임아웃 (초, 기본: 3)")
    parser.add_argument("--max-windows", type=int, default=0,
                        help="처음 N개 윈도우만 검사 (기본 0=전체)")
    parser.add_argument(
        "--led-check",
        action="store_true",
        help="각 윈도우 응답 후 LED를 확인할 때까지 일시정지")
    parser.add_argument("--csv", type=Path, default=Path("fpga_uart_results.csv"),
                        help="결과 CSV 저장 위치")
    args = parser.parse_args()

    if args.timeout <= 0 or args.max_windows < 0:
        parser.error("--timeout은 양수, --max-windows는 0 이상이어야 합니다.")

    try:
        total_steps, total_windows, inputs, expected_responses = load_vectors(args.vectors)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] Golden Model 벡터를 읽을 수 없습니다: {exc}", file=sys.stderr)
        return 2

    test_windows = (total_windows if args.max_windows == 0
                    else min(args.max_windows, total_windows))
    test_steps = test_windows + 3

    try:
        import serial  # pyserial
    except ImportError:
        print("[ERROR] pyserial이 없습니다. 터미널에서 'python -m pip install pyserial'을 실행하세요.",
              file=sys.stderr)
        return 2

    print("=" * 64)
    print(" Basys3 FPGA UART E2E Hardware Verification")
    print("=" * 64)
    print(f" Port         : {args.port}")
    print(f" UART         : {args.baud} baud, 8N1, no flow control")
    print(f" Vectors      : {args.vectors.resolve()}")
    print(f" Time steps   : {test_steps} / {total_steps}")
    print(f" Windows      : {test_windows} / {total_windows}")
    print(" Protocol     : 매 시점 4 bytes TX, 4번째 시점부터 9 bytes RX")
    print(" Note         : 각 응답 완료 후에만 다음 센서 시점 송신")
    print("=" * 64)

    passed = 0
    failed = 0
    rows = []
    start = time.perf_counter()

    try:
        with serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
            write_timeout=3.0,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        ) as port:
            # FTDI/VCP 드라이버와 FPGA가 준비될 시간을 준 뒤, 과거 RX 버퍼를 비움.
            time.sleep(0.3)
            port.reset_input_buffer()

            for step in range(test_steps):
                sensor_packet = inputs[4 * step:4 * (step + 1)]
                sent_at = time.perf_counter()
                port.write(sensor_packet)
                port.flush()

                if step < 3:  # 윈도우를 구성하기 전 첫 3개 시점은 응답 없음
                    print(f"[INIT] step {step + 1}/3, sent: {hex_bytes(sensor_packet)}")
                    continue

                window_idx = step - 3
                expected = expected_responses[9 * window_idx:9 * (window_idx + 1)]
                actual = read_exact(port, 9, args.timeout)
                elapsed_ms = (time.perf_counter() - sent_at) * 1000.0
                match = len(actual) == 9 and actual == expected

                rows.append({
                    "window": window_idx,
                    "sent_4bytes_hex": hex_bytes(sensor_packet),
                    "expected_9bytes_hex": hex_bytes(expected),
                    "received_9bytes_hex": hex_bytes(actual),
                    "pass": "PASS" if match else "FAIL",
                    "pc_roundtrip_ms": f"{elapsed_ms:.3f}",
                })

                if match:
                    passed += 1
                    if passed % 10 == 0 or window_idx == test_windows - 1:
                        print(f"[PASS] {window_idx + 1}/{test_windows} windows, "
                              f"expected=received={hex_bytes(actual)}")
                else:
                    failed += 1
                    print(f"[FAIL] window {window_idx}/{test_windows - 1}")
                    print(f"       expected: {hex_bytes(expected)}")
                    print(f"       received: {hex_bytes(actual)} (길이 {len(actual)}/9)")
                    if len(actual) < 9:
                        print("[STOP] UART 응답이 불완전합니다. 다음 윈도우 전송을 중단합니다.")
                        break
                    # 패킷 경계 손실 가능성이 있어 mismatch에서도 중단하지 않고 기록함.
                if args.led_check:
                    print(f"\n[LED CHECK] Window {window_idx + 1}")
                    print(f"UART response: {hex_bytes(actual)}")
                    input("LD0~LD3 상태를 확인하고 Enter를 누르세요...")

    except (serial.SerialException, OSError) as exc:
        print(f"[ERROR] UART 포트를 사용할 수 없습니다: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[STOP] 사용자 중단 (Ctrl+C)")
    finally:
        if rows:
            try:
                args.csv.parent.mkdir(parents=True, exist_ok=True)
                with args.csv.open("w", newline="", encoding="utf-8-sig") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[SAVED] CSV 결과: {args.csv.resolve()}")
            except OSError as exc:
                print(f"[WARN] CSV 저장 실패: {exc}")

    duration_sec = time.perf_counter() - start
    print("=" * 64)
    print(f" Hardware UART verification: {passed} PASS / {failed} FAIL "
          f"/ {test_windows} expected")
    print(f" PC test elapsed: {duration_sec:.3f} s (USB/OS 지연 포함; FPGA 처리율 아님)")
    if passed == test_windows and failed == 0:
        print(" [RESULT] ALL PASS")
        print("=" * 64)
        return 0
    print(" [RESULT] FAIL / INCOMPLETE")
    print("=" * 64)
    return 1


if __name__ == "__main__":
    sys.exit(main())
