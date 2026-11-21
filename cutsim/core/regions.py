"""CPU region semantics independent of the four native drawing clip fields."""


class ExactSurfaceDomains(tuple):
    """Complete CPU regions, with a separately cached drawing projection.

    This marker travels with the ordinary object-valued CPU domain array,
    independently of the native attribute schema and clipping policy.
    Optional ``render_domains`` is deliberately NOT used by geometry queries.
    None selects GS/GS ridge clipping; absence means an unprojected explicit
    domain, which must retain all constraints.
    """
