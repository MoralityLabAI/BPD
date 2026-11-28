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

$( BPD.EnergyFiberRank.energy_fiber_loses_at_most_one_dimension; natural arithmetic translation. $)
${
  ml-bpd-energy-loss-one.h1 $e |- A e. NN0 $.
  ml-bpd-energy-loss-one.h2 $e |- B e. NN0 $.
  ml-bpd-energy-loss-one.h3 $e |- C e. NN0 $.
  ml-bpd-energy-loss-one.h4 $e |- B <_ A $.
  ml-bpd-energy-loss-one.h5 $e |- C <_ A $.
  ml-bpd-energy-loss-one.h6 $e |- C <_ ( B + 1 ) $.
  ml-bpd-energy-loss-one $p |- ( A - B ) <_ ( ( A - C ) + 1 ) $=
    cA cB cmin co c1 cmin co cA cC cmin co cle wbr cA cB cmin co cA cC cmin co c1 caddc co cle wbr cA cB c1
    caddc co cmin co cA cB cmin co c1 cmin co cA cC cmin co cle cA cB cmin co c1 cmin co cA cB c1 caddc co
    cmin co cA cc wcel cB cc wcel c1 cc wcel cA cB cmin co c1 cmin co cA cB c1 caddc co cmin co wceq cA cr
    wcel cA cc wcel cA ml-bpd-energy-loss-one.h1 nn0rei cA recn ax-mp cB cr wcel cB cc wcel cB
    ml-bpd-energy-loss-one.h2 nn0rei cB recn ax-mp ax-1cn cA cB c1 subsub4 mp3an eqcomi cC cB c1 caddc co cle
    wbr cA cB c1 caddc co cmin co cA cC cmin co cle wbr ml-bpd-energy-loss-one.h6 cC cr wcel cB c1 caddc co
    cr wcel cA cr wcel cC cB c1 caddc co cle wbr cA cB c1 caddc co cmin co cA cC cmin co cle wbr wb cC
    ml-bpd-energy-loss-one.h3 nn0rei cB c1 cB ml-bpd-energy-loss-one.h2 nn0rei 1re readdcli cA
    ml-bpd-energy-loss-one.h1 nn0rei cC cB c1 caddc co cA lesub2 mp3an mpbi eqbrtrri cA cB cmin co c1 cA cC
    cmin co cA cB cA ml-bpd-energy-loss-one.h1 nn0rei cB ml-bpd-energy-loss-one.h2 nn0rei resubcli 1re cA cC
    cA ml-bpd-energy-loss-one.h1 nn0rei cC ml-bpd-energy-loss-one.h3 nn0rei resubcli lesubaddi mpbi
  $.
$}

$( BPD.EnergyFiberRank.delayed_rank_positive_implies_nonempty_fiber; natural arithmetic translation. $)
${
  ml-bpd-delayed-positive.h1 $e |- A e. NN0 $.
  ml-bpd-delayed-positive.h2 $e |- C e. NN0 $.
  ml-bpd-delayed-positive.h3 $e |- D e. NN0 $.
  ml-bpd-delayed-positive.h4 $e |- C <_ A $.
  ml-bpd-delayed-positive.h5 $e |- D <_ ( A - C ) $.
  ml-bpd-delayed-positive.h6 $e |- 0 < D $.
  ml-bpd-delayed-positive $p |- 0 < ( A - C ) $=
    cc0 cD clt wbr cD cA cC cmin co cle wbr cc0 cA cC cmin co clt wbr ml-bpd-delayed-positive.h6
    ml-bpd-delayed-positive.h5 cc0 cD cA cC cmin co 0re cD ml-bpd-delayed-positive.h3 nn0rei cA cC cA
    ml-bpd-delayed-positive.h1 nn0rei cC ml-bpd-delayed-positive.h2 nn0rei resubcli ltletri mp2an
  $.
$}

$( BPD.EnergyFiberRank.positive_delayed_rank_requires_spare_dimension; natural arithmetic translation. $)
${
  ml-bpd-spare-dim.h1 $e |- A e. NN0 $.
  ml-bpd-spare-dim.h2 $e |- C e. NN0 $.
  ml-bpd-spare-dim.h3 $e |- D e. NN0 $.
  ml-bpd-spare-dim.h4 $e |- C <_ A $.
  ml-bpd-spare-dim.h5 $e |- D <_ ( A - C ) $.
  ml-bpd-spare-dim.h6 $e |- 0 < D $.
  ml-bpd-spare-dim $p |- C < A $=
    cC cA clt wbr cc0 cA cC cmin co clt wbr cc0 cD clt wbr cD cA cC cmin co cle wbr cc0 cA cC cmin co clt wbr
    ml-bpd-spare-dim.h6 ml-bpd-spare-dim.h5 cc0 cD cA cC cmin co 0re cD ml-bpd-spare-dim.h3 nn0rei cA cC cA
    ml-bpd-spare-dim.h1 nn0rei cC ml-bpd-spare-dim.h2 nn0rei resubcli ltletri mp2an cC cA cC
    ml-bpd-spare-dim.h2 nn0rei cA ml-bpd-spare-dim.h1 nn0rei posdifi mpbir
  $.
$}
