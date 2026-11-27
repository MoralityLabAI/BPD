# SCCG-10: conditional stale-observer admission and gain-gap bounds

Source-only reference for the native gym proposal. No measured checkpoint
drift, training or mechanistic benefit. The mathematical guarantee is
conditional on independently established uniform bounds.

Use a common physical intervention frame with positive-definite metric G.
The cached derivative of a scalar desired response is a^T and the cached
protected-response derivative is C. At the current checkpoint suppose the
derivative errors have bounds alpha and gamma in the G-input norm:
| (a_live-a)^T u | <= alpha ||u||_G and
|| (C_live-C)u || <= gamma ||u||_G.
Assume finite nonlinear target/protected remainder bounds Rf,Rp across
the entire physical dose ball ||u||_G<=e. These must include any
measurement/rounding uncertainty relevant to the modeled response.
For current responses f(u)-f(0), g(u)-g(0), the triangle inequality gives

    f(u)-f(0) >= a^T u - E_f
    ||g(u)-g(0)|| <= ||C u||+E_p

where E_f=e alpha+Rf and E_p=e gamma+Rp.

Thus a candidate with ||C u||<=eta-E_p and target lower bound above the
registered gain floor is admissible in that mathematical setting.
If eta-E_p<0, the conservative certificate refuses. For a live
finite-dose feasible edit v, necessarily ||Cv||<=eta+E_p. For every
lambda>=0, Cauchy-Schwarz in G+lambda C^T C gives

    true optimum <=
      sqrt((e²+lambda(eta+E_p)²) a^T(G+lambda C^T C)^-1 a) + E_f.

Take the minimum across verified multipliers for a valid upper bound.
The upper bound applies only to the declared span, metric, checkpoint,
response and dose. A below-floor upper bound can screen that span,
not every action in the model. If a candidate is certified feasible,
upper minus candidate lower bounds its suboptimality over that span.
This is not a bound on training value.

For native full-suffix logits, use the actual full-suffix derivative
and a genuinely uniform finite remainder. SCCG-09's affine RMS
envelope applies to fixed-query local reader scores only; a collection
of sampled native errors, or a local RMS certificate, is not a bound
on subsequent recurrent task outputs. Use SCCG-02 tube propagation
only when all its map/domain assumptions have been certified. Absent
such bounds, use explicit native finite verification of candidates,
label prediction-only uncertainty, and fall back to current observation
or ordinary training. A passed local certificate does not establish
a beneficial optimizer step.

admission.py implements exact rational arithmetic with outward-rounded
square-root upper bounds. Positive-definite metrics and input shapes are
checked. It does not itself authenticate bounds, implement cache reuse,
or estimate runtime. Historical SCCG-01..09 stay unchanged.
