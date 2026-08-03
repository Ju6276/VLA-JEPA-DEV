import tyro

from examples.LIBERO.eval_files.libero_eval_core import (
    EvalArgs,
    eval_libero as run_eval_libero,
)


Args = EvalArgs
DEFAULT_ARGS = EvalArgs()


def eval_libero(args: EvalArgs = DEFAULT_ARGS) -> None:
    run_eval_libero(args)


if __name__ == "__main__":
    tyro.cli(eval_libero)
