from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class WireExportResult:
    requested: bool
    ok: bool
    wire_path: Path
    iges_path: Path
    converter_path: Path | None
    message: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    license_args: list[str] | None = None

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        data["wire_path"] = str(self.wire_path)
        data["iges_path"] = str(self.iges_path)
        data["converter_path"] = None if self.converter_path is None else str(self.converter_path)
        return data


def find_iges_to_al(explicit_path: str | Path | None = None) -> Path | None:
    """Find Autodesk Alias' IGES-to-wire converter.

    Alias .wire is a proprietary format, so AutoAlias intentionally converts the
    already-written IGES through Autodesk's own IGES-to-Alias tool when present.
    Older Alias documentation calls it IgesToAl; current Alias 2026 installs it
    as IgesToAlias.exe under bin/translators.
    """
    if explicit_path:
        explicit = Path(str(explicit_path).strip().strip('"')).expanduser()
        if explicit.exists():
            return explicit.resolve()

    for env_name in ("AUTOALIAS_IGES_TO_AL", "IGESTOAL_PATH", "IGES_TO_AL"):
        value = os.environ.get(env_name)
        if value:
            candidate = Path(value.strip().strip('"')).expanduser()
            if candidate.exists():
                return candidate.resolve()

    for executable in ("IgesToAlias.exe", "IgesToAl.exe", "IgesToAlias", "IgesToAl"):
        found = shutil.which(executable)
        if found:
            return Path(found).resolve()

    for candidate in _common_autodesk_candidates():
        if candidate.exists():
            return candidate.resolve()
    return None


def write_wire_from_iges(
    iges_path: str | Path,
    wire_path: str | Path,
    *,
    converter: str | Path | None = None,
    timeout_seconds: int = 240,
) -> WireExportResult:
    iges = Path(iges_path).resolve()
    wire = Path(wire_path).resolve()
    wire.parent.mkdir(parents=True, exist_ok=True)

    if not iges.exists():
        return WireExportResult(
            requested=True,
            ok=False,
            wire_path=wire,
            iges_path=iges,
            converter_path=None,
            message=f"IGES file does not exist: {iges}",
        )

    converter_path = find_iges_to_al(converter)
    if converter_path is None:
        return WireExportResult(
            requested=True,
            ok=False,
            wire_path=wire,
            iges_path=iges,
            converter_path=None,
            message=(
                "IgesToAl.exe was not found. Set AUTOALIAS_IGES_TO_AL to the full "
                "Autodesk Alias IgesToAlias.exe/IgesToAl.exe path, or add that folder to PATH."
            ),
        )

    try:
        if wire.exists():
            wire.unlink()
    except OSError:
        pass

    try:
        license_args = _alias_license_args(converter_path)
        completed = subprocess.run(
            [str(converter_path), *license_args, "-i", str(iges), "-o", str(wire)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return WireExportResult(
            requested=True,
            ok=False,
            wire_path=wire,
            iges_path=iges,
            converter_path=converter_path,
            message=f"{converter_path.name} timed out after {timeout_seconds} seconds.",
            stdout=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        )
    except OSError as exc:
        return WireExportResult(
            requested=True,
            ok=False,
            wire_path=wire,
            iges_path=iges,
            converter_path=converter_path,
            message=f"{converter_path.name} could not be started: {exc}",
        )
    parser_error = _alias_converter_reported_parser_error(completed.stdout, completed.stderr)
    ok = completed.returncode == 0 and wire.exists() and wire.stat().st_size > 0 and not parser_error
    return WireExportResult(
        requested=True,
        ok=ok,
        wire_path=wire,
        iges_path=iges,
        converter_path=converter_path,
        message=(
            "WIRE export completed."
            if ok
            else (
                f"{converter_path.name} reported IGES parser errors."
                if parser_error
                else f"{converter_path.name} did not produce a usable WIRE file."
            )
        ),
        returncode=completed.returncode,
        stdout=completed.stdout[-4000:],
        stderr=completed.stderr[-4000:],
        license_args=license_args,
    )


def write_wire_status(path: str | Path, result: WireExportResult) -> None:
    Path(path).write_text(
        json.dumps(result.to_jsonable(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _common_autodesk_candidates() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value) / "Autodesk")

    candidates: list[Path] = []
    patterns = (
        "Alias*/bin/translators/IgesToAlias.exe",
        "Alias*/bin/translators/IgesToAl.exe",
        "Alias*/*/IgesToAl.exe",
        "Alias*/IgesToAl.exe",
        "Alias*/bin/IgesToAl.exe",
        "Alias*/Utilities/IgesToAl.exe",
        "*/bin/translators/IgesToAlias.exe",
        "*/bin/translators/IgesToAl.exe",
        "*/bin/IgesToAl.exe",
        "*/IgesToAl.exe",
    )
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
    return sorted(candidates, key=lambda path: str(path).lower(), reverse=True)


def _alias_converter_reported_parser_error(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    error_markers = (
        "error at line",
        "missing delimiter",
        "iges parser",
        "parse error",
    )
    return any(marker in text for marker in error_markers)


def _alias_license_args(converter_path: Path) -> list[str]:
    explicit = os.environ.get("AUTOALIAS_ALIAS_LICENSE_ARGS")
    if explicit:
        return explicit.split()

    helper_data = _registered_alias_license()
    if helper_data:
        return _license_args_from_registration(helper_data)

    release = _alias_release_from_path(converter_path)
    product_key = {
        "2023": "966O1",
        "2026": "966R1",
    }.get(release)
    if product_key:
        return [
            "-productKey",
            product_key,
            "-productVersion",
            f"{release}.0.0.F",
            "-productLicenseType",
            "USER",
            "-productLicensePath",
            _default_named_user_license_path(),
        ]
    return []


def _registered_alias_license() -> dict[str, object] | None:
    helper = _latest_licensing_helper()
    if helper is None:
        return None
    try:
        completed = subprocess.run(
            [str(helper), "list"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            return None
        products = json.loads(completed.stdout)
    except Exception:
        return None
    aliases = [
        item
        for item in products
        if str(item.get("feature_id", "")).upper() == "ALAUST"
        and item.get("authorize_succ") is True
    ]
    if not aliases:
        return None
    return sorted(
        aliases,
        key=lambda item: str(item.get("sel_prod_ver") or item.get("def_prod_ver") or ""),
        reverse=True,
    )[0]


def _license_args_from_registration(data: dict[str, object]) -> list[str]:
    product_key = str(data.get("sel_prod_key") or data.get("def_prod_key") or "")
    product_version = str(data.get("sel_prod_ver") or data.get("def_prod_ver") or "")
    license_type = _license_type_from_registration(data)
    if not product_key or not product_version or not license_type:
        return []
    return [
        "-productKey",
        product_key,
        "-productVersion",
        product_version,
        "-productLicenseType",
        license_type,
        "-productLicensePath",
        _default_named_user_license_path(),
    ]


def _license_type_from_registration(data: dict[str, object]) -> str:
    method = data.get("lic_method")
    if method == 4 or data.get("user_lic_enabled") is True:
        return "USER"
    if method == 1:
        return "NETWORK"
    if method == 2:
        return "STANDALONE"
    return ""


def _latest_licensing_helper() -> Path | None:
    root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    base = root / "Common Files" / "Autodesk Shared" / "AdskLicensing"
    if not base.exists():
        return None
    helpers = sorted(
        base.glob("*/helper/AdskLicensingInstHelper.exe"),
        key=lambda path: str(path).lower(),
        reverse=True,
    )
    return helpers[0] if helpers else None


def _alias_release_from_path(path: Path) -> str:
    text = str(path)
    for part in Path(text).parts:
        if "AliasAutoStudio" in part:
            suffix = part.split("AliasAutoStudio", 1)[-1]
            release = suffix.split(".", 1)[0]
            if release.isdigit():
                return release
    return ""


def _default_named_user_license_path() -> str:
    program_data = os.environ.get("ProgramData", r"C:\ProgramData")
    return str(Path(program_data) / "Autodesk" / "AdskLicensingService" / "AdskLicensingService.sds")
