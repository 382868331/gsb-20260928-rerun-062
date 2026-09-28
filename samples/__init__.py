"""自带教学二进制样本（纯合成，绝不加载真实可执行程序）。"""

from .build import build_loop_sum, build_branch_call, build_unresolved_sample

__all__ = ["build_loop_sum", "build_branch_call", "build_unresolved_sample"]
