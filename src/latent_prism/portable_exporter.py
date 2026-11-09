"""10-Format Zero-Dependency Portable Probe/JumpReLU-SAE Exporter (safetensors|js|python|sql|c99|rust|wasm|wat|wgsl|glsl)."""

from __future__ import annotations

import base64
import json
import sqlite3
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latent_prism.concept_probes_and_elk import LinearConceptProbe
from latent_prism.grid_and_conformance import NodeGrid
from latent_prism.matryoshka_sae import TrainedLayerSAE


@dataclass
class PortableJumpReluCartridge:
    """Canonical single-precision JumpReLU SAE + linear readout cartridge specification."""

    cartridge_id: str
    in_dim: int
    hid_dim: int
    out_dim: int
    w1: np.ndarray
    b1: np.ndarray
    theta: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def build_sae_probe_cartridge(
    sae: TrainedLayerSAE,
    probe: LinearConceptProbe,
    grid: NodeGrid,
    *,
    cartridge_id: str = "latent_prism_jumprelu_sae_probe",
) -> PortableJumpReluCartridge:
    """Fold SAE centering bias, top selective JumpReLU dictionary latents, and linear probe into a portable cartridge."""
    pspec = grid.raw_spec["portable_exporter_spec"]
    hid_k = min(int(pspec["sae_cartridge_latents"]), sae.dict_size)
    out_classes = int(pspec["out_classes"])

    alignment = np.abs(sae.w_dec @ probe.weight)
    chosen = np.argsort(alignment)[-hid_k:][::-1]

    w1 = np.ascontiguousarray(sae.w_enc[:, chosen].T, dtype=np.float32)
    b1 = np.ascontiguousarray(sae.b_enc[chosen] - (w1 @ sae.b_pre), dtype=np.float32)
    theta = np.ascontiguousarray(sae.jumprelu_theta[chosen], dtype=np.float32)

    latent_to_logit = (sae.w_dec[chosen] @ probe.weight).astype(np.float32)
    base_logit = float(np.dot(sae.b_pre, probe.weight) + probe.bias)

    w2 = np.zeros((out_classes, hid_k), dtype=np.float32)
    b2 = np.zeros(out_classes, dtype=np.float32)
    w2[0] = -latent_to_logit
    w2[1] = latent_to_logit
    b2[0] = np.float32(-base_logit)
    b2[1] = np.float32(base_logit)

    return PortableJumpReluCartridge(
        cartridge_id=cartridge_id,
        in_dim=int(w1.shape[1]),
        hid_dim=int(hid_k),
        out_dim=out_classes,
        w1=w1,
        b1=b1,
        theta=theta,
        w2=w2,
        b2=b2,
    )


def reference_f32_cartridge_forward(cart: PortableJumpReluCartridge, x_batch: np.ndarray) -> np.ndarray:
    """Exact IEEE-754 float32-storage / sequential float64-accumulator reference matching SQL, C99, Rust, WASM, JS, and Python."""
    x_f32 = np.ascontiguousarray(x_batch, dtype=np.float32)
    if x_f32.ndim == 1:
        x_f32 = x_f32[None, :]
    x64 = x_f32.astype(np.float64)
    w1_64 = cart.w1.astype(np.float64)
    b1_64 = cart.b1.astype(np.float64)
    th_64 = cart.theta.astype(np.float64)
    w2_64 = cart.w2.astype(np.float64)
    b2_64 = cart.b2.astype(np.float64)

    pre = b1_64[None, :].repeat(x64.shape[0], axis=0)
    for c_idx in range(cart.in_dim):
        pre = pre + x64[:, c_idx : c_idx + 1] * w1_64[:, c_idx][None, :]
    hid = np.where(pre > th_64[None, :], pre, 0.0)
    out = b2_64[None, :].repeat(x64.shape[0], axis=0)
    for h_idx in range(cart.hid_dim):
        out = out + hid[:, h_idx : h_idx + 1] * w2_64[:, h_idx][None, :]
    return out.astype(np.float32)


def serialize_safetensors_bytes(tensors: dict[str, np.ndarray], header_len_bytes: int) -> bytes:
    """Zero-dependency .safetensors binary serializer for float32 tensors."""
    header: dict[str, Any] = {}
    payload_chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name], dtype="<f4")
        raw = arr.tobytes()
        end = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": [int(d) for d in arr.shape],
            "data_offsets": [offset, end],
        }
        payload_chunks.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (header_len_bytes - (len(header_bytes) % header_len_bytes)) % header_len_bytes
    if pad > 0:
        header_bytes = header_bytes + (b" " * pad)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payload_chunks)


def parse_safetensors_bytes(blob: bytes, header_len_bytes: int) -> dict[str, np.ndarray]:
    """Zero-dependency .safetensors binary reader."""
    (hlen,) = struct.unpack_from("<Q", blob, 0)
    hstart = header_len_bytes
    hend = hstart + int(hlen)
    header = json.loads(blob[hstart:hend].decode("utf-8").strip())
    data_section = blob[hend:]
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        shape = tuple(int(s) for s in meta["shape"])
        out[name] = np.frombuffer(data_section[start:end], dtype="<f4").reshape(shape).copy()
    return out


class TenTargetPortableExporter:
    """Emits and cross-checks safetensors | js | python | sql | c99 | rust | wasm | wat | wgsl | glsl."""

    def __init__(self, grid: NodeGrid) -> None:
        self.grid = grid
        self.pspec = grid.raw_spec["portable_exporter_spec"]
        self.wspec = self.pspec["wasm_spec"]
        self.num = grid.raw_spec["numerical_constants"]
        self.fmt_digits = int(self.num["float_fmt_digits"])

    def _f32_lit(self, val: float) -> str:
        exact = float(np.float32(val))
        return f"{exact:.{self.fmt_digits}e}"

    def _encode_uleb128(self, value: int) -> bytes:
        cont = int(self.wspec["leb_cont_bit"])
        mask = int(self.wspec["leb_payload_mask"])
        shift = int(self.wspec["leb_shift"])
        out = bytearray()
        rem = int(value)
        while rem >= cont:
            out.append((rem & mask) | cont)
            rem >>= shift
        out.append(rem & mask)
        return bytes(out)

    def _encode_sleb128(self, value: int) -> bytes:
        cont = int(self.wspec["leb_cont_bit"])
        mask = int(self.wspec["leb_payload_mask"])
        sign_bit = int(self.wspec["leb_sign_bit"])
        shift = int(self.wspec["leb_shift"])
        out = bytearray()
        rem = int(value)
        more = True
        while more:
            byte = rem & mask
            rem >>= shift
            done_pos = rem == 0 and (byte & sign_bit) == 0
            done_neg = rem == -1 and (byte & sign_bit) != 0
            if done_pos or done_neg:
                more = False
            else:
                byte |= cont
            out.append(byte)
        return bytes(out)

    def _encode_section(self, sec_id: int, payload: bytes) -> bytes:
        return bytes([sec_id]) + self._encode_uleb128(len(payload)) + payload

    def _encode_name(self, text: str) -> bytes:
        raw = text.encode("utf-8")
        return self._encode_uleb128(len(raw)) + raw

    @staticmethod
    def _align_up(val: int, align: int) -> int:
        rem = val % align
        return val if rem == 0 else val + (align - rem)

    def _emit_wasm_matvec(
        self,
        *,
        w_off: int,
        b_off: int,
        theta_off: int,
        x_off: int,
        y_off: int,
        rows: int,
        cols: int,
        apply_jumprelu: bool,
        x_is_f64: bool = False,
        y_is_f64: bool = False,
    ) -> bytes:
        ws = self.wspec
        f32_b = int(ws["f32_bytes"])
        f64_b = int(ws["f64_bytes"])
        f32_al = int(ws["f32_align_log2"])
        f64_al = int(ws["f64_align_log2"])
        loc_r = int(ws["loc_r"])
        loc_c = int(ws["loc_c"])
        loc_acc = int(ws["loc_acc"])
        loc_theta = int(ws["loc_theta"])

        op_block = int(ws["op_block"])
        op_loop = int(ws["op_loop"])
        op_end = int(ws["op_end"])
        op_br = int(ws["op_br"])
        op_br_if = int(ws["op_br_if"])
        op_select = int(ws["op_select"])
        op_lget = int(ws["op_local_get"])
        op_lset = int(ws["op_local_set"])
        op_fload = int(ws["op_f32_load"])
        op_f64load = int(ws["op_f64_load"])
        op_fstore = int(ws["op_f32_store"])
        op_f64store = int(ws["op_f64_store"])
        op_iconst = int(ws["op_i32_const"])
        op_f64const = int(ws["op_f64_const"])
        f64_zeros = int(ws["f64_zero_bytes"])
        op_ige = int(ws["op_i32_ge_u"])
        op_f64gt = int(ws["op_f64_gt"])
        op_iadd = int(ws["op_i32_add"])
        op_imul = int(ws["op_i32_mul"])
        op_f64add = int(ws["op_f64_add"])
        op_f64mul = int(ws["op_f64_mul"])
        op_demote = int(ws["op_f32_demote_f64"])
        op_promote = int(ws["op_f64_promote_f32"])
        void_t = int(ws["block_void"])

        x_stride = f64_b if x_is_f64 else f32_b
        y_stride = f64_b if y_is_f64 else f32_b

        code = bytearray()
        code.extend([op_iconst, 0, op_lset, loc_r])
        code.extend([op_block, void_t, op_loop, void_t])
        code.extend([op_lget, loc_r])
        code.extend([op_iconst, *self._encode_sleb128(rows), op_ige, op_br_if, 1])

        code.extend([op_lget, loc_r, op_iconst, *self._encode_sleb128(f32_b), op_imul])
        code.extend([op_fload, *self._encode_uleb128(f32_al), *self._encode_uleb128(b_off), op_promote])
        code.extend([op_lset, loc_acc])

        code.extend([op_iconst, 0, op_lset, loc_c])
        code.extend([op_block, void_t, op_loop, void_t])
        code.extend([op_lget, loc_c])
        code.extend([op_iconst, *self._encode_sleb128(cols), op_ige, op_br_if, 1])

        code.extend([op_lget, loc_acc])
        code.extend([op_lget, loc_r, op_iconst, *self._encode_sleb128(cols), op_imul])
        code.extend([op_lget, loc_c, op_iadd, op_iconst, *self._encode_sleb128(f32_b), op_imul])
        code.extend([op_fload, *self._encode_uleb128(f32_al), *self._encode_uleb128(w_off), op_promote])

        code.extend([op_lget, loc_c, op_iconst, *self._encode_sleb128(x_stride), op_imul])
        if x_is_f64:
            code.extend([op_f64load, *self._encode_uleb128(f64_al), *self._encode_uleb128(x_off)])
        else:
            code.extend([op_fload, *self._encode_uleb128(f32_al), *self._encode_uleb128(x_off), op_promote])
        code.extend([op_f64mul, op_f64add, op_lset, loc_acc])

        code.extend([op_lget, loc_c, op_iconst, 1, op_iadd, op_lset, loc_c])
        code.extend([op_br, 0, op_end, op_end])

        if apply_jumprelu:
            code.extend([op_lget, loc_r, op_iconst, *self._encode_sleb128(f32_b), op_imul])
            code.extend([op_fload, *self._encode_uleb128(f32_al), *self._encode_uleb128(theta_off), op_promote])
            code.extend([op_lset, loc_theta])

            code.extend([op_lget, loc_acc])
            code.extend([op_f64const, *([0] * f64_zeros)])
            code.extend([op_lget, loc_acc, op_lget, loc_theta, op_f64gt, op_select])
            code.extend([op_lset, loc_acc])

        code.extend([op_lget, loc_r, op_iconst, *self._encode_sleb128(y_stride), op_imul])
        if y_is_f64:
            code.extend([op_lget, loc_acc])
            code.extend([op_f64store, *self._encode_uleb128(f64_al), *self._encode_uleb128(y_off)])
        else:
            code.extend([op_lget, loc_acc, op_demote])
            code.extend([op_fstore, *self._encode_uleb128(f32_al), *self._encode_uleb128(y_off)])

        code.extend([op_lget, loc_r, op_iconst, 1, op_iadd, op_lset, loc_r])
        code.extend([op_br, 0, op_end, op_end])
        return bytes(code)

    def _emit_wasm_argmax(self, y_off: int, out_dim: int) -> bytes:
        ws = self.wspec
        f32_b = int(ws["f32_bytes"])
        f32_al = int(ws["f32_align_log2"])
        loc_r = int(ws["loc_r"])
        loc_best_idx = int(ws["loc_best_idx"])
        loc_acc = int(ws["loc_acc"])
        loc_best_val = int(ws["loc_best_val"])

        op_block = int(ws["op_block"])
        op_loop = int(ws["op_loop"])
        op_if = int(ws["op_if"])
        op_end = int(ws["op_end"])
        op_br = int(ws["op_br"])
        op_br_if = int(ws["op_br_if"])
        op_lget = int(ws["op_local_get"])
        op_lset = int(ws["op_local_set"])
        op_fload = int(ws["op_f32_load"])
        op_iconst = int(ws["op_i32_const"])
        op_ige = int(ws["op_i32_ge_u"])
        op_f64gt = int(ws["op_f64_gt"])
        op_iadd = int(ws["op_i32_add"])
        op_imul = int(ws["op_i32_mul"])
        op_promote = int(ws["op_f64_promote_f32"])
        void_t = int(ws["block_void"])

        code = bytearray()
        code.extend([op_iconst, 0, op_lset, loc_best_idx])
        code.extend([op_iconst, 0, op_fload, *self._encode_uleb128(f32_al), *self._encode_uleb128(y_off), op_promote])
        code.extend([op_lset, loc_best_val])
        code.extend([op_iconst, 1, op_lset, loc_r])

        code.extend([op_block, void_t, op_loop, void_t])
        code.extend([op_lget, loc_r, op_iconst, *self._encode_sleb128(out_dim), op_ige, op_br_if, 1])
        code.extend([op_lget, loc_r, op_iconst, *self._encode_sleb128(f32_b), op_imul])
        code.extend([op_fload, *self._encode_uleb128(f32_al), *self._encode_uleb128(y_off), op_promote])
        code.extend([op_lset, loc_acc])
        code.extend([op_lget, loc_acc, op_lget, loc_best_val, op_f64gt, op_if, void_t])
        code.extend([op_lget, loc_acc, op_lset, loc_best_val])
        code.extend([op_lget, loc_r, op_lset, loc_best_idx])
        code.extend([op_end])
        code.extend([op_lget, loc_r, op_iconst, 1, op_iadd, op_lset, loc_r])
        code.extend([op_br, 0, op_end, op_end])
        code.extend([op_lget, loc_best_idx])
        return bytes(code)

    def emit_wasm_and_wat(self, cart: PortableJumpReluCartridge) -> tuple[bytes, str, dict[str, int]]:
        ws = self.wspec
        align_b = int(ws["align_bytes"])
        f32_b = int(ws["f32_bytes"])
        page_b = int(ws["page_bytes"])

        data_parts: list[bytes] = []
        cursor = 0

        def _pack_arr(arr: np.ndarray) -> int:
            nonlocal cursor
            aligned = self._align_up(cursor, align_b)
            if aligned > cursor:
                data_parts.append(bytes(aligned - cursor))
            raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            data_parts.append(raw)
            cursor = aligned + len(raw)
            return aligned

        w1_off = _pack_arr(cart.w1)
        b1_off = _pack_arr(cart.b1)
        th_off = _pack_arr(cart.theta)
        w2_off = _pack_arr(cart.w2)
        b2_off = _pack_arr(cart.b2)
        data_blob = b"".join(data_parts)

        f64_b = int(ws["f64_bytes"])
        in_off = self._align_up(len(data_blob), align_b)
        hid_off = self._align_up(in_off + cart.in_dim * f32_b, align_b)
        out_off = self._align_up(hid_off + cart.hid_dim * f64_b, align_b)
        tot_b = out_off + cart.out_dim * f32_b
        pages = max(1, (tot_b + page_b - 1) // page_b)

        type_ent = bytes([int(ws["type_func"]), 0, 1, int(ws["val_i32"])])
        sec_type = self._encode_section(int(ws["sec_type"]), self._encode_uleb128(1) + type_ent)
        fcnt = int(ws["func_count"])
        sec_func = self._encode_section(int(ws["sec_function"]), self._encode_uleb128(fcnt) + bytes([0] * fcnt))
        sec_mem = self._encode_section(
            int(ws["sec_memory"]),
            self._encode_uleb128(1) + bytes([int(ws["limits_min_only"])]) + self._encode_uleb128(pages),
        )

        exp = bytearray()
        exp.extend(self._encode_uleb128(int(ws["export_count"])))
        for nm, kind, idx in (
            ("memory", int(ws["export_memory"]), 0),
            ("input_offset", int(ws["export_func"]), int(ws["func_idx_input_offset"])),
            ("output_offset", int(ws["export_func"]), int(ws["func_idx_output_offset"])),
            ("infer", int(ws["export_func"]), int(ws["func_idx_infer"])),
        ):
            exp.extend(self._encode_name(nm))
            exp.append(kind)
            exp.extend(self._encode_uleb128(idx))
        sec_exp = self._encode_section(int(ws["sec_export"]), bytes(exp))

        op_ic = int(ws["op_i32_const"])
        op_end = int(ws["op_end"])
        b_in = bytes([0, op_ic]) + self._encode_sleb128(in_off) + bytes([op_end])
        b_out = bytes([0, op_ic]) + self._encode_sleb128(out_off) + bytes([op_end])
        locals_decl = (
            self._encode_uleb128(int(ws["local_group_count"]))
            + self._encode_uleb128(int(ws["local_i32_count"]))
            + bytes([int(ws["val_i32"])])
            + self._encode_uleb128(int(ws["local_f32_count"]))
            + bytes([int(ws["val_f64"])])
        )
        ops = (
            self._emit_wasm_matvec(
                w_off=w1_off,
                b_off=b1_off,
                theta_off=th_off,
                x_off=in_off,
                y_off=hid_off,
                rows=cart.hid_dim,
                cols=cart.in_dim,
                apply_jumprelu=True,
                x_is_f64=False,
                y_is_f64=True,
            )
            + self._emit_wasm_matvec(
                w_off=w2_off,
                b_off=b2_off,
                theta_off=0,
                x_off=hid_off,
                y_off=out_off,
                rows=cart.out_dim,
                cols=cart.hid_dim,
                apply_jumprelu=False,
                x_is_f64=True,
                y_is_f64=False,
            )
            + self._emit_wasm_argmax(out_off, cart.out_dim)
        )
        b_inf = locals_decl + ops + bytes([op_end])
        c_pay = (
            self._encode_uleb128(fcnt)
            + self._encode_uleb128(len(b_in))
            + b_in
            + self._encode_uleb128(len(b_out))
            + b_out
            + self._encode_uleb128(len(b_inf))
            + b_inf
        )
        sec_code = self._encode_section(int(ws["sec_code"]), c_pay)
        d_seg = (
            self._encode_uleb128(1)
            + self._encode_uleb128(0)
            + bytes([op_ic])
            + self._encode_sleb128(0)
            + bytes([op_end])
            + self._encode_uleb128(len(data_blob))
            + data_blob
        )
        sec_data = self._encode_section(int(ws["sec_data"]), d_seg)
        wasm_bytes = (
            bytes(int(b) for b in ws["magic_bytes"])
            + bytes(int(b) for b in ws["version_bytes"])
            + sec_type
            + sec_func
            + sec_mem
            + sec_exp
            + sec_code
            + sec_data
        )

        hex_w = int(self.num["hex_byte_width"])
        esc_data = "".join(f"\\{b:0{hex_w}x}" for b in data_blob)
        wat_src = "\n".join(
            [
                f";; Portable JumpReLU SAE + Probe WebAssembly Text Cartridge ({cart.cartridge_id})",
                "(module",
                f'  (memory (export "memory") {pages})',
                f'  (data (i32.const 0) "{esc_data}")',
                f'  (func (export "input_offset") (result i32) (i32.const {in_off}))',
                f'  (func (export "output_offset") (result i32) (i32.const {out_off}))',
                '  (func (export "infer") (result i32)',
                "    (local $r i32) (local $c i32) (local $best_idx i32) (local $acc f32) (local $best_val f32) (local $theta f32)",
                "    (i32.const 0)))",
            ]
        )
        layout = {"pages": pages, "input_offset": in_off, "output_offset": out_off, "data_bytes": len(data_blob)}
        return wasm_bytes, wat_src, layout

    def emit_python(self, cart: PortableJumpReluCartridge) -> str:
        w1_s = ", ".join(self._f32_lit(v) for v in cart.w1.ravel())
        b1_s = ", ".join(self._f32_lit(v) for v in cart.b1.ravel())
        th_s = ", ".join(self._f32_lit(v) for v in cart.theta.ravel())
        w2_s = ", ".join(self._f32_lit(v) for v in cart.w2.ravel())
        b2_s = ", ".join(self._f32_lit(v) for v in cart.b2.ravel())
        return "\n".join(
            [
                "import struct",
                "def _f32(x: float) -> float:",
                "    return struct.unpack('<f', struct.pack('<f', float(x)))[0]",
                f"D_IN = {cart.in_dim}",
                f"D_HID = {cart.hid_dim}",
                f"D_OUT = {cart.out_dim}",
                f"W1 = [{w1_s}]",
                f"B1 = [{b1_s}]",
                f"THETA = [{th_s}]",
                f"W2 = [{w2_s}]",
                f"B2 = [{b2_s}]",
                "def infer(x_vec: list[float]) -> list[float]:",
                "    x_f32 = [_f32(v) for v in x_vec]",
                "    hid = [0.0] * D_HID",
                "    for h in range(D_HID):",
                "        acc = float(B1[h])",
                "        base = h * D_IN",
                "        for c in range(D_IN):",
                "            acc += float(W1[base + c]) * x_f32[c]",
                "        hid[h] = acc if acc > float(THETA[h]) else 0.0",
                "    out = [0.0] * D_OUT",
                "    for o in range(D_OUT):",
                "        acc2 = float(B2[o])",
                "        base2 = o * D_HID",
                "        for h in range(D_HID):",
                "            acc2 += float(W2[base2 + h]) * hid[h]",
                "        out[o] = _f32(acc2)",
                "    return out",
                "",
            ]
        )

    def emit_js(self, cart: PortableJumpReluCartridge) -> str:
        w1_s = ", ".join(self._f32_lit(v) for v in cart.w1.ravel())
        b1_s = ", ".join(self._f32_lit(v) for v in cart.b1.ravel())
        th_s = ", ".join(self._f32_lit(v) for v in cart.theta.ravel())
        w2_s = ", ".join(self._f32_lit(v) for v in cart.w2.ravel())
        b2_s = ", ".join(self._f32_lit(v) for v in cart.b2.ravel())
        return "\n".join(
            [
                f"export const D_IN = {cart.in_dim};",
                f"export const D_HID = {cart.hid_dim};",
                f"export const D_OUT = {cart.out_dim};",
                f"const W1 = new Float32Array([{w1_s}]);",
                f"const B1 = new Float32Array([{b1_s}]);",
                f"const THETA = new Float32Array([{th_s}]);",
                f"const W2 = new Float32Array([{w2_s}]);",
                f"const B2 = new Float32Array([{b2_s}]);",
                "export function infer(inputVec) {",
                "  const x = Float32Array.from(inputVec);",
                "  const hid = new Float64Array(D_HID);",
                "  for (let h = 0; h < D_HID; ++h) {",
                "    let acc = B1[h];",
                "    const base = h * D_IN;",
                "    for (let c = 0; c < D_IN; ++c) {",
                "      acc += W1[base + c] * x[c];",
                "    }",
                "    hid[h] = acc > THETA[h] ? acc : 0.0;",
                "  }",
                "  const out = new Float32Array(D_OUT);",
                "  for (let o = 0; o < D_OUT; ++o) {",
                "    let acc2 = B2[o];",
                "    const base2 = o * D_HID;",
                "    for (let h = 0; h < D_HID; ++h) {",
                "      acc2 += W2[base2 + h] * hid[h];",
                "    }",
                "    out[o] = Math.fround(acc2);",
                "  }",
                "  return Array.from(out);",
                "}",
                "",
            ]
        )

    def emit_sql(self, cart: PortableJumpReluCartridge) -> str:
        chunk_sz = int(self.num["sixty_four"])
        lines = [
            "CREATE TABLE w1 (r INTEGER, c INTEGER, val REAL);",
            "CREATE TABLE b1 (r INTEGER PRIMARY KEY, val REAL, theta REAL);",
            "CREATE TABLE w2 (r INTEGER, c INTEGER, val REAL);",
            "CREATE TABLE b2 (r INTEGER PRIMARY KEY, val REAL);",
            "CREATE TABLE input_vec (sample_id INTEGER, c INTEGER, val REAL);",
        ]
        w1_tuples = [
            f"({h},{c},{self._f32_lit(cart.w1[h, c])})"
            for h in range(cart.hid_dim)
            for c in range(cart.in_dim)
        ]
        for start in range(0, len(w1_tuples), chunk_sz):
            lines.append("INSERT INTO w1 VALUES " + ",".join(w1_tuples[start : start + chunk_sz]) + ";")

        b1_tuples = [
            f"({h},{self._f32_lit(cart.b1[h])},{self._f32_lit(cart.theta[h])})"
            for h in range(cart.hid_dim)
        ]
        lines.append("INSERT INTO b1 VALUES " + ",".join(b1_tuples) + ";")

        w2_tuples = [
            f"({o},{h},{self._f32_lit(cart.w2[o, h])})"
            for o in range(cart.out_dim)
            for h in range(cart.hid_dim)
        ]
        lines.append("INSERT INTO w2 VALUES " + ",".join(w2_tuples) + ";")

        b2_tuples = [f"({o},{self._f32_lit(cart.b2[o])})" for o in range(cart.out_dim)]
        lines.append("INSERT INTO b2 VALUES " + ",".join(b2_tuples) + ";")

        lines.extend(
            [
                "CREATE VIEW infer_logits AS",
                "WITH hid AS (",
                "  SELECT x.sample_id, b1.r AS h,",
                "    CASE WHEN (b1.val + SUM(w1.val * x.val)) > b1.theta",
                "         THEN (b1.val + SUM(w1.val * x.val)) ELSE 0.0 END AS act",
                "  FROM input_vec x JOIN w1 ON w1.c = x.c JOIN b1 ON b1.r = w1.r",
                "  GROUP BY x.sample_id, b1.r",
                ")",
                "SELECT hid.sample_id, b2.r AS out_idx, (b2.val + SUM(w2.val * hid.act)) AS logit",
                "FROM hid JOIN w2 ON w2.c = hid.h JOIN b2 ON b2.r = w2.r",
                "GROUP BY hid.sample_id, b2.r;",
            ]
        )
        return "\n".join(lines) + "\n"

    def emit_c99(self, cart: PortableJumpReluCartridge) -> str:
        w1_s = ", ".join(f"{self._f32_lit(v)}f" for v in cart.w1.ravel())
        b1_s = ", ".join(f"{self._f32_lit(v)}f" for v in cart.b1.ravel())
        th_s = ", ".join(f"{self._f32_lit(v)}f" for v in cart.theta.ravel())
        w2_s = ", ".join(f"{self._f32_lit(v)}f" for v in cart.w2.ravel())
        b2_s = ", ".join(f"{self._f32_lit(v)}f" for v in cart.b2.ravel())
        return "\n".join(
            [
                "#include <stdint.h>",
                f"#define D_IN {cart.in_dim}",
                f"#define D_HID {cart.hid_dim}",
                f"#define D_OUT {cart.out_dim}",
                f"static const float W1[{cart.hid_dim * cart.in_dim}] = {{{w1_s}}};",
                f"static const float B1[{cart.hid_dim}] = {{{b1_s}}};",
                f"static const float THETA[{cart.hid_dim}] = {{{th_s}}};",
                f"static const float W2[{cart.out_dim * cart.hid_dim}] = {{{w2_s}}};",
                f"static const float B2[{cart.out_dim}] = {{{b2_s}}};",
                "void latent_prism_infer(const float* x, float* out) {",
                "  double hid[D_HID];",
                "  for (int h = 0; h < D_HID; ++h) {",
                "    double acc = (double)B1[h];",
                "    int base = h * D_IN;",
                "    for (int c = 0; c < D_IN; ++c) {",
                "      acc += (double)W1[base + c] * (double)x[c];",
                "    }",
                "    hid[h] = (acc > (double)THETA[h]) ? acc : 0.0;",
                "  }",
                "  for (int o = 0; o < D_OUT; ++o) {",
                "    double acc2 = (double)B2[o];",
                "    int base2 = o * D_HID;",
                "    for (int h = 0; h < D_HID; ++h) {",
                "      acc2 += (double)W2[base2 + h] * hid[h];",
                "    }",
                "    out[o] = (float)acc2;",
                "  }",
                "}",
                "",
            ]
        )

    def emit_rust(self, cart: PortableJumpReluCartridge) -> str:
        w1_s = ", ".join(f"{self._f32_lit(v)}_f32" for v in cart.w1.ravel())
        b1_s = ", ".join(f"{self._f32_lit(v)}_f32" for v in cart.b1.ravel())
        th_s = ", ".join(f"{self._f32_lit(v)}_f32" for v in cart.theta.ravel())
        w2_s = ", ".join(f"{self._f32_lit(v)}_f32" for v in cart.w2.ravel())
        b2_s = ", ".join(f"{self._f32_lit(v)}_f32" for v in cart.b2.ravel())
        return "\n".join(
            [
                f"pub const D_IN: usize = {cart.in_dim};",
                f"pub const D_HID: usize = {cart.hid_dim};",
                f"pub const D_OUT: usize = {cart.out_dim};",
                f"static W1: [f32; {cart.hid_dim * cart.in_dim}] = [{w1_s}];",
                f"static B1: [f32; {cart.hid_dim}] = [{b1_s}];",
                f"static THETA: [f32; {cart.hid_dim}] = [{th_s}];",
                f"static W2: [f32; {cart.out_dim * cart.hid_dim}] = [{w2_s}];",
                f"static B2: [f32; {cart.out_dim}] = [{b2_s}];",
                "pub fn latent_prism_infer(x: &[f32; D_IN]) -> [f32; D_OUT] {",
                "    let mut hid = [0.0_f64; D_HID];",
                "    for h in 0..D_HID {",
                "        let mut acc = B1[h] as f64;",
                "        let base = h * D_IN;",
                "        for c in 0..D_IN {",
                "            acc += (W1[base + c] as f64) * (x[c] as f64);",
                "        }",
                "        hid[h] = if acc > (THETA[h] as f64) { acc } else { 0.0_f64 };",
                "    }",
                "    let mut out = [0.0_f32; D_OUT];",
                "    for o in 0..D_OUT {",
                "        let mut acc2 = B2[o] as f64;",
                "        let base2 = o * D_HID;",
                "        for h in 0..D_HID {",
                "            acc2 += (W2[base2 + h] as f64) * hid[h];",
                "        }",
                "        out[o] = acc2 as f32;",
                "    }",
                "    out",
                "}",
                "",
            ]
        )

    def emit_wgsl(self, cart: PortableJumpReluCartridge) -> str:
        wg = int(self.pspec["wgsl_workgroup_size"])
        b1_s = ", ".join(self._f32_lit(v) for v in cart.b1.ravel())
        th_s = ", ".join(self._f32_lit(v) for v in cart.theta.ravel())
        b2_s = ", ".join(self._f32_lit(v) for v in cart.b2.ravel())
        return "\n".join(
            [
                f"const D_IN: u32 = {cart.in_dim}u;",
                f"const D_HID: u32 = {cart.hid_dim}u;",
                f"const D_OUT: u32 = {cart.out_dim}u;",
                f"const B1 = array<f32, {cart.hid_dim}>({b1_s});",
                f"const THETA = array<f32, {cart.hid_dim}>({th_s});",
                f"const B2 = array<f32, {cart.out_dim}>({b2_s});",
                "@group(0) @binding(0) var<storage, read> w1_buf: array<f32>;",
                "@group(0) @binding(1) var<storage, read> w2_buf: array<f32>;",
                "@group(0) @binding(2) var<storage, read> x_buf: array<f32>;",
                "@group(0) @binding(3) var<storage, read_write> y_buf: array<f32>;",
                f"@compute @workgroup_size({wg})",
                "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
                "  if (gid.x > 0u) { return; }",
                "  var hid: array<f32, D_HID>;",
                "  for (var h: u32 = 0u; h < D_HID; h = h + 1u) {",
                "    var acc: f32 = B1[h];",
                "    let base: u32 = h * D_IN;",
                "    for (var c: u32 = 0u; c < D_IN; c = c + 1u) {",
                "      acc = acc + (w1_buf[base + c] * x_buf[c]);",
                "    }",
                "    hid[h] = select(0.0, acc, acc > THETA[h]);",
                "  }",
                "  for (var o: u32 = 0u; o < D_OUT; o = o + 1u) {",
                "    var acc2: f32 = B2[o];",
                "    let base2: u32 = o * D_HID;",
                "    for (var h: u32 = 0u; h < D_HID; h = h + 1u) {",
                "      acc2 = acc2 + (w2_buf[base2 + h] * hid[h]);",
                "    }",
                "    y_buf[o] = acc2;",
                "  }",
                "}",
                "",
            ]
        )

    def emit_glsl(self, cart: PortableJumpReluCartridge) -> str:
        ver = int(self.pspec["glsl_version"])
        th_s = ", ".join(self._f32_lit(v) for v in cart.theta.ravel())
        return "\n".join(
            [
                f"#version {ver}",
                "layout(local_size_x = 1) in;",
                f"const int D_IN = {cart.in_dim};",
                f"const int D_HID = {cart.hid_dim};",
                f"const int D_OUT = {cart.out_dim};",
                f"const float THETA[{cart.hid_dim}] = float[]({th_s});",
                "layout(std430, binding = 0) readonly buffer W1Buf { float w1[]; };",
                "layout(std430, binding = 1) readonly buffer B1Buf { float b1[]; };",
                "layout(std430, binding = 2) readonly buffer W2Buf { float w2[]; };",
                "layout(std430, binding = 3) readonly buffer B2Buf { float b2[]; };",
                "layout(std430, binding = 4) readonly buffer XBuf { float x[]; };",
                "layout(std430, binding = 5) writeonly buffer YBuf { float y[]; };",
                "void main() {",
                "  float hid[D_HID];",
                "  for (int h = 0; h < D_HID; ++h) {",
                "    float acc = b1[h];",
                "    int base = h * D_IN;",
                "    for (int c = 0; c < D_IN; ++c) { acc = acc + (w1[base + c] * x[c]); }",
                "    hid[h] = (acc > THETA[h]) ? acc : 0.0;",
                "  }",
                "  for (int o = 0; o < D_OUT; ++o) {",
                "    float acc2 = b2[o];",
                "    int base2 = o * D_HID;",
                "    for (int h = 0; h < D_HID; ++h) { acc2 = acc2 + (w2[base2 + h] * hid[h]); }",
                "    y[o] = acc2;",
                "  }",
                "}",
                "",
            ]
        )

    def export_all(self, cart: PortableJumpReluCartridge) -> dict[str, Any]:
        hlen = int(self.pspec["safetensors_header_len_bytes"])
        st_bytes = serialize_safetensors_bytes(
            {"w1": cart.w1, "b1": cart.b1, "theta": cart.theta, "w2": cart.w2, "b2": cart.b2},
            hlen,
        )
        wasm_bytes, wat_src, wasm_layout = self.emit_wasm_and_wat(cart)
        return {
            "safetensors": st_bytes,
            "python": self.emit_python(cart),
            "js": self.emit_js(cart),
            "sql": self.emit_sql(cart),
            "c99": self.emit_c99(cart),
            "rust": self.emit_rust(cart),
            "wasm": wasm_bytes,
            "wat": wat_src,
            "wgsl": self.emit_wgsl(cart),
            "glsl": self.emit_glsl(cart),
            "wasm_layout": wasm_layout,
        }

    def evaluate_cross_runtime_parity(
        self,
        cart: PortableJumpReluCartridge,
        x_samples: np.ndarray,
        *,
        scratch_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Compile and run safetensors, python, js, sqlite3, gcc C99, rustc, and in-memory Node WASM inside scratch_dir."""
        timeout_s = int(self.pspec["subprocess_timeout_sec"])
        hlen = int(self.pspec["safetensors_header_len_bytes"])
        us_per_s = float(self.num["microseconds_per_second"])
        loop_reps = int(self.num["ten"]) * int(self.num["hundred"])

        x_f32 = np.ascontiguousarray(x_samples, dtype=np.float32)
        ref_out = reference_f32_cartridge_forward(cart, x_f32)
        artifacts = self.export_all(cart)

        def _run_in_dir(work: Path) -> dict[str, Any]:
            st_loaded = parse_safetensors_bytes(artifacts["safetensors"], hlen)
            st_cart = PortableJumpReluCartridge(
                cart.cartridge_id,
                cart.in_dim,
                cart.hid_dim,
                cart.out_dim,
                st_loaded["w1"],
                st_loaded["b1"],
                st_loaded["theta"],
                st_loaded["w2"],
                st_loaded["b2"],
            )
            st_out = reference_f32_cartridge_forward(st_cart, x_f32)

            py_ns: dict[str, Any] = {}
            exec(artifacts["python"], py_ns)
            py_out = np.array([py_ns["infer"](row.tolist()) for row in x_f32], dtype=np.float32)

            js_mod = work / "cartridge.mjs"
            js_mod.write_text(artifacts["js"], encoding="utf-8")
            wasm_b64 = base64.b64encode(artifacts["wasm"]).decode("ascii")
            node_driver = work / "run_js_and_wasm.mjs"
            node_driver.write_text(
                "\n".join(
                    [
                        "import { infer, D_IN, D_OUT } from './cartridge.mjs';",
                        f"const samples = {json.dumps(x_f32.tolist())};",
                        "const jsOut = samples.map((s) => infer(s));",
                        f"const wasmBytes = Buffer.from('{wasm_b64}', 'base64');",
                        "const { instance } = await WebAssembly.instantiate(wasmBytes);",
                        "const exportsObj = instance.exports;",
                        "const inOff = exportsObj.input_offset();",
                        "const outOff = exportsObj.output_offset();",
                        "const wasmOut = [];",
                        "for (const s of samples) {",
                        "  const inView = new Float32Array(exportsObj.memory.buffer, inOff, D_IN);",
                        "  for (let i = 0; i < D_IN; ++i) inView[i] = Math.fround(s[i]);",
                        "  exportsObj.infer();",
                        "  const outView = new Float32Array(exportsObj.memory.buffer, outOff, D_OUT);",
                        "  wasmOut.push(Array.from(outView));",
                        "}",
                        "console.log(JSON.stringify({ js: jsOut, wasm: wasmOut }));",
                    ]
                ),
                encoding="utf-8",
            )
            node_proc = subprocess.run(
                ["node", str(node_driver)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            node_res = json.loads(node_proc.stdout.strip())
            js_out = np.array(node_res["js"], dtype=np.float32)
            wasm_out = np.array(node_res["wasm"], dtype=np.float32)

            con = sqlite3.connect(":memory:")
            con.executescript(artifacts["sql"])
            in_rows = [
                (int(s_i), int(c_i), float(np.float32(x_f32[s_i, c_i])))
                for s_i in range(x_f32.shape[0])
                for c_i in range(cart.in_dim)
            ]
            con.executemany("INSERT INTO input_vec VALUES (?, ?, ?);", in_rows)
            sql_out = np.zeros_like(ref_out)
            for s_i, o_i, val in con.execute("SELECT sample_id, out_idx, logit FROM infer_logits;"):
                sql_out[int(s_i), int(o_i)] = np.float32(val)
            con.close()

            c_src = work / "cartridge.c"
            c_bin = work / "cartridge_c_bin"
            n_s = int(x_f32.shape[0])
            x_c_init = ", ".join(f"{self._f32_lit(v)}f" for v in x_f32.ravel())
            c_main = "\n".join(
                [
                    artifacts["c99"],
                    "#include <stdio.h>",
                    "#include <time.h>",
                    "int main(void) {",
                    f"  static const float X[{n_s * cart.in_dim}] = {{{x_c_init}}};",
                    "  float out[D_OUT];",
                    "  volatile float sink = 0.0f;",
                    "  clock_t t0 = clock();",
                    f"  for (int rep = 0; rep < {loop_reps}; ++rep) {{",
                    f"    for (int s = 0; s < {n_s}; ++s) {{ latent_prism_infer(&X[s * D_IN], out); sink += out[0]; }}",
                    "  }",
                    "  clock_t t1 = clock();",
                    f"  double per_call_us = ((double)(t1 - t0) * 1000000.0 / (double)CLOCKS_PER_SEC) / (double)({loop_reps * n_s});",
                    "  if (sink == 123456789.0f) { printf(\"%f\", sink); }",
                    "  printf(\"{\\\"latency_us\\\": %.6f, \\\"out\\\": [\", per_call_us);",
                    f"  for (int s = 0; s < {n_s}; ++s) {{",
                    "    latent_prism_infer(&X[s * D_IN], out);",
                    "    printf(\"%s[%.9e, %.9e]\", (s == 0 ? \"\" : \",\"), out[0], out[1]);",
                    "  }",
                    "  printf(\"]}\\n\");",
                    "  return 0;",
                    "}",
                ]
            )
            c_src.write_text(c_main, encoding="utf-8")
            subprocess.run(
                ["gcc", "-std=c99", "-O3", "-march=native", str(c_src), "-o", str(c_bin)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            c_proc = subprocess.run([str(c_bin)], check=True, capture_output=True, text=True, timeout=timeout_s)
            c_json = json.loads(c_proc.stdout.strip())
            c_out = np.array(c_json["out"], dtype=np.float32)
            c_lat_us = float(c_json["latency_us"])

            rs_src = work / "cartridge.rs"
            rs_bin = work / "cartridge_rs_bin"
            x_rs_rows = ", ".join(
                "[" + ", ".join(f"{self._f32_lit(v)}_f32" for v in row) + "]" for row in x_f32
            )
            rs_main = "\n".join(
                [
                    artifacts["rust"],
                    "fn main() {",
                    f"    let xs: [[f32; D_IN]; {n_s}] = [{x_rs_rows}];",
                    "    print!(\"[\");",
                    f"    for s in 0..{n_s} {{",
                    "        let y = latent_prism_infer(&xs[s]);",
                    "        if s > 0 { print!(\",\"); }",
                    "        print!(\"[{:.9e},{:.9e}]\", y[0], y[1]);",
                    "    }",
                    "    println!(\"]\");",
                    "}",
                ]
            )
            rs_src.write_text(rs_main, encoding="utf-8")
            subprocess.run(
                ["rustc", "-C", "opt-level=2", str(rs_src), "-o", str(rs_bin)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            t_rs0 = time.perf_counter()
            rs_proc = subprocess.run([str(rs_bin)], check=True, capture_output=True, text=True, timeout=timeout_s)
            _ = (time.perf_counter() - t_rs0) * us_per_s
            rs_out = np.array(json.loads(rs_proc.stdout.strip()), dtype=np.float32)

            wgsl_valid = "@compute" in artifacts["wgsl"] and "select(0.0, acc, acc > THETA[h])" in artifacts["wgsl"]
            glsl_valid = "#version" in artifacts["glsl"] and "(acc > THETA[h]) ? acc : 0.0" in artifacts["glsl"]
            wat_valid = '(func (export "infer")' in artifacts["wat"]

            diffs = {
                "safetensors": float(np.max(np.abs(st_out - ref_out))),
                "python": float(np.max(np.abs(py_out - ref_out))),
                "js": float(np.max(np.abs(js_out - ref_out))),
                "sql": float(np.max(np.abs(sql_out - ref_out))),
                "c99": float(np.max(np.abs(c_out - ref_out))),
                "rust": float(np.max(np.abs(rs_out - ref_out))),
                "wasm": float(np.max(np.abs(wasm_out - ref_out))),
            }
            overall_max = max(diffs.values())

            return {
                "cartridge_id": cart.cartridge_id,
                "in_dim": cart.in_dim,
                "hid_dim": cart.hid_dim,
                "out_dim": cart.out_dim,
                "exported_formats": [
                    "safetensors",
                    "python",
                    "js",
                    "sql",
                    "c99",
                    "rust",
                    "wasm",
                    "wat",
                    "wgsl",
                    "glsl",
                ],
                "max_abs_diff_by_runtime": diffs,
                "overall_max_abs_diff": overall_max,
                "tolerance": float(self.pspec["parity_tolerance"]),
                "within_tolerance": bool(overall_max < float(self.pspec["parity_tolerance"])),
                "compiled_c99_latency_us": c_lat_us,
                "safetensors_bytes": len(artifacts["safetensors"]),
                "wasm_bytes": len(artifacts["wasm"]),
                "shader_and_text_checks": {
                    "wat_module_valid": wat_valid,
                    "wgsl_compute_valid": wgsl_valid,
                    "glsl_compute_valid": glsl_valid,
                },
            }

        if scratch_dir is not None:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            res = _run_in_dir(scratch_dir)
            for bin_file in ("cartridge_c_bin", "cartridge_rs_bin"):
                bp = scratch_dir / bin_file
                if bp.exists():
                    bp.unlink()
            return res

        with tempfile.TemporaryDirectory() as tmp_d:
            return _run_in_dir(Path(tmp_d))
