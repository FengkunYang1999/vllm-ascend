# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the consolidated engine-core platform patch.

``patch_engine_core.py`` is the single entry-point patch for
``EngineCoreProc.run_engine_core``. What is guarded here (everything reachable
from CPU UT):

* the consolidated patch is applied eagerly at import: the live
  ``EngineCoreProc.run_engine_core`` is ``_run_engine_core_patch_func``;
* ``_OriginalRunEngineCore`` stashes the pristine upstream ``run_engine_core``
  -- the genuine original the wrapper must delegate to, regardless of import
  ordering;
* ``_patch_dp_engine_core_proc`` selects ``DyntraLBDPEngineCoreProc`` when
  dyntra-lb is enabled, else leaves the module-global ``DPEngineCoreProc``
  untouched; the swap is deferred until the engine-core entry point runs;
* ``_run_engine_core_patch_func`` initializes the ascend config and
  delegates to the upstream entry point with ``dp_rank`` / ``local_dp_rank``
  passed through;
* ``_apply_patch`` is idempotent; child process re-applies the profiling
  patch exactly through this import chain when unpickling the
  ``run_engine_core`` wrapper).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.engine.core import DPEngineCoreProc as _UpstreamDPEngineCoreProc
from vllm.v1.engine.core import EngineCoreProc as _UpstreamEngineCoreProc

import vllm_ascend.patch.platform.patch_engine_core as _engine_core_patch
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc


def _ascend_config():
    return SimpleNamespace(scheduler_config=SimpleNamespace())


def _dyntra_config(enabled: bool, enable_diagnostics: bool = True):
    return SimpleNamespace(enabled=enabled, enable_diagnostics=enable_diagnostics)


# ---------------------------------------------------------------------------
# 1. the consolidated patch took effect at import
# ---------------------------------------------------------------------------


def test_engine_core_patch_applied_at_import():
    assert _engine_core_patch._PATCHED is True
    assert _UpstreamEngineCoreProc.run_engine_core is _engine_core_patch._run_engine_core_patch_func
    assert _engine_core_mod.DPEngineCoreProc is _UpstreamDPEngineCoreProc, (
        "DPEngineCoreProc must stay upstream until the engine-core entry point runs"
    )


def test_original_run_engine_core_stashes_pristine_upstream():
    """The wrapper must delegate to the genuine upstream ``run_engine_core``.

    ``_OriginalRunEngineCore`` is bound to ``EngineCoreProc.run_engine_core``
    at the patch module's first import -- before the staticmethod rebind line
    runs -- so it is the pristine upstream original regardless of import
    ordering. The live class attribute is already the patch wrapper by test
    time and cannot be used for this check."""
    assert _engine_core_patch._OriginalRunEngineCore is not _engine_core_patch._run_engine_core_patch_func
    src = inspect.getsource(_engine_core_patch._OriginalRunEngineCore)
    assert "DPEngineCoreProc(" in src, (
        "upstream run_engine_core no longer instantiates DPEngineCoreProc by "
        "module-global name; the _engine_core_mod.DPEngineCoreProc swap in "
        "patch_engine_core.py would silently break."
    )


def test_apply_patch_is_idempotent():
    _engine_core_patch._apply_patch()

    assert _UpstreamEngineCoreProc.run_engine_core is _engine_core_patch._run_engine_core_patch_func


def test_dp_proc_swap_uses_dyntra_when_enabled(monkeypatch):
    monkeypatch.setattr(_engine_core_patch, "_get_dyntra_lb_config", lambda _c: _dyntra_config(True))
    print_mock = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "dyntra_print_rank_0", print_mock)
    # Register restore of the module-global swap the code under test performs.
    monkeypatch.setattr(_engine_core_mod, "DPEngineCoreProc", _engine_core_mod.DPEngineCoreProc)

    _engine_core_patch._patch_dp_engine_core_proc(vllm_config=object(), dp_rank=2)

    assert _engine_core_mod.DPEngineCoreProc is DyntraLBDPEngineCoreProc
    print_mock.assert_called_once_with("Enable DyntraLB DP load balancing.", 2, True)


def test_dp_proc_swap_leaves_pristine_when_dyntra_disabled(monkeypatch):
    """Other scheduler paths, including recompute, keep the upstream proc."""
    monkeypatch.setattr(_engine_core_patch, "_get_dyntra_lb_config", lambda _c: _dyntra_config(False))
    monkeypatch.setattr(_engine_core_mod, "DPEngineCoreProc", _engine_core_mod.DPEngineCoreProc)

    _engine_core_patch._patch_dp_engine_core_proc(vllm_config=object(), dp_rank=0)

    assert _engine_core_mod.DPEngineCoreProc is _UpstreamDPEngineCoreProc


# ---------------------------------------------------------------------------
# 3. the run_engine_core wrapper
# ---------------------------------------------------------------------------


def test_run_engine_core_patch_delegates_and_passes_rank_args(monkeypatch):
    expected = object()
    calls = []

    def original(*args, dp_rank=0, local_dp_rank=0, **kwargs):
        calls.append((args, dp_rank, local_dp_rank, kwargs))
        return expected

    monkeypatch.setattr(_engine_core_patch, "_OriginalRunEngineCore", original)
    init_ascend = MagicMock(return_value=_ascend_config())
    monkeypatch.setattr(_engine_core_patch, "init_ascend_config", init_ascend)
    dp_proc_patch = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "_patch_dp_engine_core_proc", dp_proc_patch)

    vllm_config = object()
    result = _engine_core_patch._run_engine_core_patch_func(
        "engine-arg",
        vllm_config=vllm_config,
        dp_rank=1,
        local_dp_rank=2,
    )

    assert result is expected
    assert calls == [
        (
            ("engine-arg",),
            1,
            2,
            {"vllm_config": vllm_config},
        )
    ]
    init_ascend.assert_called_once_with(vllm_config)
    dp_proc_patch.assert_called_once_with(vllm_config, 1)
