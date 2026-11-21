import argparse
from pathlib import Path

import numpy as np

from lsj_biometrics.config.operating_point import PACKAGED, load_operating_point
from lsj_biometrics.gallery.enrollment import EnrollmentResult
from lsj_biometrics.gallery.identification import IdentificationResult
from lsj_biometrics.gallery.verification import VerificationResult
from lsj_biometrics.system import BiometricSystem, build_system, enroll_folder, photo


def main() -> None:
    from lsj_biometrics.training.commands import register

    parser = argparse.ArgumentParser(prog="lsj")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_system_commands(subparsers)
    register(subparsers)
    args = parser.parse_args()
    args.func(args)


def add_system_commands(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("enroll-folder", help="enrol every photo of a folder, the identity from its file name without _YYYY")
    command.add_argument("folder", type=Path)
    command.add_argument("--config", type=Path, default=PACKAGED)
    command.set_defaults(func=run_enroll_folder)

    for name, handler in (("inscribir", run_inscribir), ("inscribir_sd", run_inscribir_sd), ("verificar", run_verificar), ("verificar_sd", run_verificar_sd)):
        command = subparsers.add_parser(name)
        command.add_argument("--id", required=True)
        command.add_argument("--foto", type=Path, required=True)
        command.add_argument("--config", type=Path, default=PACKAGED)
        command.set_defaults(func=handler)

    for name, handler in (("identificar", run_identificar), ("identificar_sd", run_identificar_sd)):
        command = subparsers.add_parser(name)
        command.add_argument("--foto", type=Path, required=True)
        command.add_argument("--config", type=Path, default=PACKAGED)
        command.set_defaults(func=handler)

    for name, handler in (("inscribir_t", run_inscribir_t), ("verificar_t", run_verificar_t)):
        command = subparsers.add_parser(name)
        command.add_argument("--id", required=True)
        command.add_argument("--template", type=Path, required=True, help="a .npy file with one vector")
        command.add_argument("--config", type=Path, default=PACKAGED)
        command.set_defaults(func=handler)

    command = subparsers.add_parser("identificar_t")
    command.add_argument("--template", type=Path, required=True, help="a .npy file with one vector")
    command.add_argument("--config", type=Path, default=PACKAGED)
    command.set_defaults(func=run_identificar_t)


def system_of(config: Path) -> BiometricSystem:
    return build_system(load_operating_point(config))


def template_file(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def run_enroll_folder(args: argparse.Namespace) -> None:
    results, failed = enroll_folder(system_of(args.config), args.folder)
    for result in results:
        print_enrollment(result)
    print(f"[OK] enrolled {len(results)} of {len(results) + len(failed)} photos from {args.folder}")
    if failed:
        raise SystemExit(1)


def run_inscribir(args: argparse.Namespace) -> None:
    print_enrollment(system_of(args.config).inscribir(args.id, photo(args.foto)))


def run_inscribir_sd(args: argparse.Namespace) -> None:
    print_enrollment(system_of(args.config).inscribir_sd(args.id, photo(args.foto)))


def run_inscribir_t(args: argparse.Namespace) -> None:
    print_enrollment(system_of(args.config).inscribir_t(args.id, template_file(args.template)))


def run_verificar(args: argparse.Namespace) -> None:
    print_verification(system_of(args.config).verificar(args.id, photo(args.foto)))


def run_verificar_sd(args: argparse.Namespace) -> None:
    print_verification(system_of(args.config).verificar_sd(args.id, photo(args.foto)))


def run_verificar_t(args: argparse.Namespace) -> None:
    print_verification(system_of(args.config).verificar_t(args.id, template_file(args.template)))


def run_identificar(args: argparse.Namespace) -> None:
    print_identification(system_of(args.config).identificar(photo(args.foto)))


def run_identificar_sd(args: argparse.Namespace) -> None:
    print_identification(system_of(args.config).identificar_sd(photo(args.foto)))


def run_identificar_t(args: argparse.Namespace) -> None:
    print_identification(system_of(args.config).identificar_t(template_file(args.template)))


def print_enrollment(result: EnrollmentResult) -> None:
    for template in result.templates:
        print(f"[OK] enrolled {result.identity.identity_id}: template {template.template_id}")


def print_verification(result: VerificationResult) -> None:
    verdict = "rejected"
    if result.accepted:
        verdict = "accepted"
    print(f"[OK] {result.identity_id}: {verdict}, score {result.score:.4f}")


def print_identification(result: IdentificationResult) -> None:
    if result.decision is None:
        print("[OK] no enrolled identity at or above the threshold")
    else:
        print(f"[OK] identified {result.decision.identity_id}, score {result.decision.score:.4f}")
    for position, match in enumerate(result.matches, start=1):
        print(f"  {position}. {match.identity_id} {match.score:.4f}")
