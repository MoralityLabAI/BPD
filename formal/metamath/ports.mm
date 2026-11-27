$( BPD.OptimalBasis.conjugate_mask_iff_commute; pointwise equational translation. $)
${
  ml-bpd-mask-commute.h1 $e |- ( G ` ( F ` B ) ) = B $.
  ml-bpd-mask-commute.h2 $e |- ( F ` ( G ` A ) ) = A $.
  ml-bpd-mask-commute $p |- ( ( G ` A ) = B <-> A = ( F ` B ) ) $=
    cA cG cfv cB wceq cA cB cF cfv wceq cA cG cfv cB wceq cA cG cfv cF cfv cB cF cfv wceq cA cB cF cfv wceq
    cA cG cfv cB cF fveq2 cA cG cfv cF cfv cA cB cF cfv ml-bpd-mask-commute.h2 eqeq1i sylib cA cB cF cfv wceq
    cA cG cfv cB cF cfv cG cfv wceq cA cG cfv cB wceq cA cB cF cfv cG fveq2 cB cF cfv cG cfv cB cA cG cfv
    ml-bpd-mask-commute.h1 eqeq2i sylib impbii
  $.
$}

$( BPD.OptimalBasis.block_activity_zero_iff_of_commute; pointwise equational translation. $)
${
  ml-bpd-block-zero.h1 $e |- A = ( F ` B ) $.
  ml-bpd-block-zero.h2 $e |- ( G ` ( F ` B ) ) = B $.
  ml-bpd-block-zero.h3 $e |- ( F ` C ) = C $.
  ml-bpd-block-zero.h4 $e |- ( G ` C ) = C $.
  ml-bpd-block-zero $p |- ( A = C <-> B = C ) $=
    cA cC wceq cB cC wceq cA cC wceq cB cC cG cfv wceq cB cC wceq cA cC wceq cB cF cfv cG cfv cC cG cfv wceq
    cB cC cG cfv wceq cA cC wceq cB cF cfv cC wceq cB cF cfv cG cfv cC cG cfv wceq cA cC wceq cB cF cfv cC
    wceq cA cB cF cfv cC ml-bpd-block-zero.h1 eqeq1i biimpi cB cF cfv cC cG fveq2 syl cB cF cfv cG cfv cC cG
    cfv wceq cB cC cG cfv wceq cB cF cfv cG cfv cB cC cG cfv ml-bpd-block-zero.h2 eqeq1i biimpi syl cB cC cG
    cfv wceq cB cC wceq cC cG cfv cC cB ml-bpd-block-zero.h4 eqeq2i biimpi syl cB cC wceq cB cF cfv cC wceq
    cA cC wceq cB cC wceq cB cF cfv cC cF cfv wceq cB cF cfv cC wceq cB cC cF fveq2 cB cF cfv cC cF cfv wceq
    cB cF cfv cC wceq cC cF cfv cC cB cF cfv ml-bpd-block-zero.h3 eqeq2i biimpi syl cA cC wceq cB cF cfv cC
    wceq cA cB cF cfv cC ml-bpd-block-zero.h1 eqeq1i biimpri syl impbii
  $.
$}
