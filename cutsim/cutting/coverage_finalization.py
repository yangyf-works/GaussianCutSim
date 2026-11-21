"""Coverage-disabled diagnostics for the production calculation."""
import numpy as np


def skipped_coverage(*, scope='unverified_stage4_output'):
    """Explicitly unverified diagnostics; disabled never means sampled success."""
    stats = dict(coverage_enabled=False, coverage_status='disabled', coverage_initial_status='disabled',
                 coverage_audit_stage='disabled', coverage_final_audited=False,
                 coverage_audit_count=0, coverage_repair_passes=0,
                 coverage_repair_rays=0, coverage_repair_added=0,
                 coverage_repair_published=0, coverage_initial_missing_tool=0,
                 coverage_ms=0.)
    packet = dict(stats=stats.copy(), audit_stage='disabled',
                  points=np.empty((0, 3)), normals=np.empty((0, 3)),
                  missing=np.empty((0, 2), bool), source_rows=np.empty(0, np.int64),
                  tool_rows=np.empty(0, np.int64), omitted=0,
                  scope=scope)
    return stats, packet
