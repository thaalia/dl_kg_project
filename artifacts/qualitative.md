# Qualitative predictions on FB15k-237

_Sampled 5 test triples (seed=42). Top-10 candidates per side, filtered against the train+val+test triples._

## Triple 1: (/m/0btyf5z, /film/film/release_date_s./film/film_regional_release_date/film_release_region, /m/0345h)

### artifacts/custom/RotatE_random/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **9**
  - `/m/0jgd` (score=-6.271)
  - `/m/035qy` (score=-6.282)
  - `/m/05b4w` (score=-6.330)
  - `/m/03rj0` (score=-6.335)
  - `/m/06mkj` (score=-6.344)
  - `/m/03_3d` (score=-6.404)
  - `/m/0chghy` (score=-6.410)
  - `/m/0154j` (score=-6.414)
  - `/m/0345h` (score=-6.415) ←gold
  - `/m/05v8c` (score=-6.419)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **19**
  - `/m/09gmmt6` (score=-5.844)
  - `/m/0gh65c5` (score=-5.950)
  - `/m/049w1q` (score=-5.952)
  - `/m/03yvf2` (score=-6.000)
  - `/m/0m491` (score=-6.117)
  - `/m/01xlqd` (score=-6.183)
  - `/m/0gvvm6l` (score=-6.195)
  - `/m/01shy7` (score=-6.252)
  - `/m/08j7lh` (score=-6.286)
  - `/m/02pxst` (score=-6.304)

### artifacts/custom/RotatE_hard/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **1**
  - `/m/0345h` (score=-7.358) ←gold
  - `/m/06mkj` (score=-7.408)
  - `/m/03_3d` (score=-7.445)
  - `/m/0jgd` (score=-7.478)
  - `/m/05qhw` (score=-7.497)
  - `/m/015fr` (score=-7.509)
  - `/m/0chghy` (score=-7.519)
  - `/m/035qy` (score=-7.520)
  - `/m/05b4w` (score=-7.574)
  - `/m/0154j` (score=-7.603)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **19**
  - `/m/03yvf2` (score=-6.776)
  - `/m/0m491` (score=-6.850)
  - `/m/0gvvm6l` (score=-6.860)
  - `/m/049w1q` (score=-6.866)
  - `/m/0gh65c5` (score=-6.948)
  - `/m/0g5879y` (score=-6.966)
  - `/m/01shy7` (score=-7.092)
  - `/m/09gmmt6` (score=-7.098)
  - `/m/0gvvf4j` (score=-7.122)
  - `/m/04yg13l` (score=-7.179)

### artifacts/custom/RotatE_mixed_30_70/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **1**
  - `/m/0345h` (score=-7.429) ←gold
  - `/m/035qy` (score=-7.527)
  - `/m/06mkj` (score=-7.605)
  - `/m/09c7w0` (score=-7.661)
  - `/m/0154j` (score=-7.665)
  - `/m/0jgd` (score=-7.672)
  - `/m/03rt9` (score=-7.685)
  - `/m/03_3d` (score=-7.741)
  - `/m/0chghy` (score=-7.769)
  - `/m/05b4w` (score=-7.831)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **26**
  - `/m/049w1q` (score=-6.727)
  - `/m/0gvvm6l` (score=-6.809)
  - `/m/0m491` (score=-6.913)
  - `/m/0gh65c5` (score=-6.937)
  - `/m/09gmmt6` (score=-6.991)
  - `/m/0g5879y` (score=-7.001)
  - `/m/03yvf2` (score=-7.003)
  - `/m/01shy7` (score=-7.132)
  - `/m/0184tc` (score=-7.137)
  - `/m/032clf` (score=-7.237)

### artifacts/custom/RotatE_mixed_50_50/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **3**
  - `/m/035qy` (score=-7.594)
  - `/m/06mkj` (score=-7.632)
  - `/m/0345h` (score=-7.675) ←gold
  - `/m/0jgd` (score=-7.714)
  - `/m/05b4w` (score=-7.750)
  - `/m/03rj0` (score=-7.805)
  - `/m/015fr` (score=-7.813)
  - `/m/09c7w0` (score=-7.816)
  - `/m/05qhw` (score=-7.846)
  - `/m/0b90_r` (score=-7.855)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **45**
  - `/m/0gh65c5` (score=-6.824)
  - `/m/03yvf2` (score=-6.946)
  - `/m/049w1q` (score=-6.977)
  - `/m/0gvvm6l` (score=-6.997)
  - `/m/0g5879y` (score=-7.079)
  - `/m/01shy7` (score=-7.085)
  - `/m/09gmmt6` (score=-7.183)
  - `/m/0m491` (score=-7.234)
  - `/m/0gvvf4j` (score=-7.268)
  - `/m/0c8tkt` (score=-7.269)

### artifacts/custom/RotatE_mixed_70_30/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **3**
  - `/m/06mkj` (score=-7.278)
  - `/m/0jgd` (score=-7.397)
  - `/m/0345h` (score=-7.426) ←gold
  - `/m/05qhw` (score=-7.458)
  - `/m/0chghy` (score=-7.501)
  - `/m/035qy` (score=-7.533)
  - `/m/03_3d` (score=-7.574)
  - `/m/05b4w` (score=-7.600)
  - `/m/09c7w0` (score=-7.624)
  - `/m/0154j` (score=-7.656)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **15**
  - `/m/0gh65c5` (score=-6.868)
  - `/m/03yvf2` (score=-6.944)
  - `/m/0gvvm6l` (score=-6.952)
  - `/m/01shy7` (score=-7.190)
  - `/m/049w1q` (score=-7.202)
  - `/m/09gmmt6` (score=-7.211)
  - `/m/0m491` (score=-7.242)
  - `/m/0184tc` (score=-7.242)
  - `/m/0g5879y` (score=-7.249)
  - `/m/0mbql` (score=-7.335)

## Triple 2: (/m/03ryks, /music/artist/track_contributions./music/track_contribution/role, /m/013y1f)

### artifacts/custom/RotatE_random/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **3**
  - `/m/026t6` (score=-5.499)
  - `/m/042v_gx` (score=-5.632)
  - `/m/013y1f` (score=-5.710) ←gold
  - `/m/0342h` (score=-5.743)
  - `/m/05148p4` (score=-5.841)
  - `/m/01vj9c` (score=-5.859)
  - `/m/03qjg` (score=-5.898)
  - `/m/03gvt` (score=-6.025)
  - `/m/07brj` (score=-6.138)
  - `/m/07y_7` (score=-6.164)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **2**
  - `/m/01vsyg9` (score=-5.669)
  - `/m/03ryks` (score=-5.710) ←gold
  - `/m/01p95y0` (score=-5.748)
  - `/m/01wsl7c` (score=-5.836)
  - `/m/0565cz` (score=-5.850)
  - `/m/023l9y` (score=-5.944)
  - `/m/01lvcs1` (score=-6.023)
  - `/m/018gkb` (score=-6.035)
  - `/m/0140t7` (score=-6.035)
  - `/m/01mwsnc` (score=-6.049)

### artifacts/custom/RotatE_hard/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **4**
  - `/m/0342h` (score=-6.951)
  - `/m/042v_gx` (score=-6.984)
  - `/m/026t6` (score=-7.086)
  - `/m/013y1f` (score=-7.109) ←gold
  - `/m/05148p4` (score=-7.265)
  - `/m/03qjg` (score=-7.368)
  - `/m/03gvt` (score=-7.489)
  - `/m/01vj9c` (score=-7.500)
  - `/m/07brj` (score=-7.671)
  - `/m/01s0ps` (score=-7.726)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **2**
  - `/m/01vsyg9` (score=-7.073)
  - `/m/03ryks` (score=-7.109) ←gold
  - `/m/01p95y0` (score=-7.211)
  - `/m/023slg` (score=-7.237)
  - `/m/07r4c` (score=-7.240)
  - `/m/01vrnsk` (score=-7.252)
  - `/m/01wsl7c` (score=-7.277)
  - `/m/01l4g5` (score=-7.305)
  - `/m/03j24kf` (score=-7.308)
  - `/m/0lsw9` (score=-7.313)

### artifacts/custom/RotatE_mixed_30_70/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **4**
  - `/m/0342h` (score=-6.670)
  - `/m/042v_gx` (score=-6.694)
  - `/m/026t6` (score=-6.764)
  - `/m/013y1f` (score=-6.799) ←gold
  - `/m/05148p4` (score=-6.963)
  - `/m/03qjg` (score=-7.055)
  - `/m/01vj9c` (score=-7.243)
  - `/m/03gvt` (score=-7.426)
  - `/m/07brj` (score=-7.488)
  - `/m/0gkd1` (score=-7.657)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **1**
  - `/m/03ryks` (score=-6.799) ←gold
  - `/m/01vsyg9` (score=-7.034)
  - `/m/01p95y0` (score=-7.208)
  - `/m/023l9y` (score=-7.210)
  - `/m/023slg` (score=-7.245)
  - `/m/0lsw9` (score=-7.276)
  - `/m/0140t7` (score=-7.295)
  - `/m/082brv` (score=-7.308)
  - `/m/01lvcs1` (score=-7.329)
  - `/m/018gkb` (score=-7.333)

### artifacts/custom/RotatE_mixed_50_50/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **4**
  - `/m/026t6` (score=-6.977)
  - `/m/0342h` (score=-7.112)
  - `/m/042v_gx` (score=-7.169)
  - `/m/013y1f` (score=-7.249) ←gold
  - `/m/05148p4` (score=-7.322)
  - `/m/01vj9c` (score=-7.442)
  - `/m/03qjg` (score=-7.556)
  - `/m/07brj` (score=-7.558)
  - `/m/03gvt` (score=-7.645)
  - `/m/01s0ps` (score=-7.917)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **6**
  - `/m/01vsyg9` (score=-7.012)
  - `/m/023l9y` (score=-7.121)
  - `/m/0140t7` (score=-7.121)
  - `/m/023slg` (score=-7.219)
  - `/m/01vrnsk` (score=-7.231)
  - `/m/03ryks` (score=-7.249) ←gold
  - `/m/01mxt_` (score=-7.250)
  - `/m/032t2z` (score=-7.253)
  - `/m/0lsw9` (score=-7.289)
  - `/m/07r4c` (score=-7.333)

### artifacts/custom/RotatE_mixed_70_30/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **6**
  - `/m/0342h` (score=-6.987)
  - `/m/042v_gx` (score=-7.171)
  - `/m/026t6` (score=-7.173)
  - `/m/05148p4` (score=-7.228)
  - `/m/03qjg` (score=-7.265)
  - `/m/013y1f` (score=-7.344) ←gold
  - `/m/01vj9c` (score=-7.532)
  - `/m/03gvt` (score=-7.690)
  - `/m/07brj` (score=-7.753)
  - `/m/07y_7` (score=-8.114)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **10**
  - `/m/01p95y0` (score=-7.061)
  - `/m/01vrnsk` (score=-7.175)
  - `/m/01vsyg9` (score=-7.175)
  - `/m/0140t7` (score=-7.187)
  - `/m/023l9y` (score=-7.217)
  - `/m/023slg` (score=-7.238)
  - `/m/03j24kf` (score=-7.249)
  - `/m/01vsksr` (score=-7.330)
  - `/m/032t2z` (score=-7.331)
  - `/m/03ryks` (score=-7.344) ←gold

## Triple 3: (/m/07b_l, /location/location/contains, /m/013m4v)

### artifacts/custom/RotatE_random/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **1032**
  - `/m/08mbj5d` (score=-6.904)
  - `/m/01n7q` (score=-7.330)
  - `/m/030qb3t` (score=-7.372)
  - `/m/02_286` (score=-7.420)
  - `/m/04n6k` (score=-7.515)
  - `/m/09nqf` (score=-7.536)
  - `/m/02jx1` (score=-7.623)
  - `/m/02dtg` (score=-7.637)
  - `/m/03ksy` (score=-7.655)
  - `/m/0cwx_` (score=-7.657)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **17**
  - `/m/08mbj5d` (score=-8.276)
  - `/m/04n6k` (score=-8.333)
  - `/m/02jx1` (score=-8.518)
  - `/m/01n7q` (score=-8.537)
  - `/m/04ztj` (score=-8.654)
  - `/m/07ssc` (score=-8.685)
  - `/m/02hrh1q` (score=-8.730)
  - `/m/05zppz` (score=-8.784)
  - `/m/01d_h8` (score=-8.816)
  - `/m/0dxtg` (score=-8.829)

### artifacts/custom/RotatE_hard/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **491**
  - `/m/08mbj5d` (score=-7.951)
  - `/m/0n7q7` (score=-8.009)
  - `/m/0fw2y` (score=-8.179)
  - `/m/0trv` (score=-8.373)
  - `/m/013f9v` (score=-8.421)
  - `/m/030qb3t` (score=-8.432)
  - `/m/01n7q` (score=-8.446)
  - `/m/02dtg` (score=-8.461)
  - `/m/029qzx` (score=-8.471)
  - `/m/0dyl9` (score=-8.483)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **6**
  - `/m/01n7q` (score=-8.864)
  - `/m/07ssc` (score=-8.998)
  - `/m/08mbj5d` (score=-9.021)
  - `/m/059rby` (score=-9.119)
  - `/m/04n6k` (score=-9.141)
  - `/m/07b_l` (score=-9.210) ←gold
  - `/m/02jx1` (score=-9.224)
  - `/m/0345h` (score=-9.268)
  - `/m/0f8l9c` (score=-9.285)
  - `/m/0d060g` (score=-9.292)

### artifacts/custom/RotatE_mixed_30_70/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **412**
  - `/m/08mbj5d` (score=-7.979)
  - `/m/0n7q7` (score=-8.094)
  - `/m/0fw2y` (score=-8.221)
  - `/m/02dtg` (score=-8.377)
  - `/m/030qb3t` (score=-8.379)
  - `/m/0nr_q` (score=-8.383)
  - `/m/02_286` (score=-8.431)
  - `/m/01n7q` (score=-8.455)
  - `/m/01jzyx` (score=-8.459)
  - `/m/0sv6n` (score=-8.472)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **4**
  - `/m/08mbj5d` (score=-8.963)
  - `/m/01n7q` (score=-9.025)
  - `/m/07ssc` (score=-9.037)
  - `/m/07b_l` (score=-9.106) ←gold
  - `/m/059rby` (score=-9.121)
  - `/m/02jx1` (score=-9.215)
  - `/m/0345h` (score=-9.255)
  - `/m/02_286` (score=-9.293)
  - `/m/03rjj` (score=-9.302)
  - `/m/04ztj` (score=-9.334)

### artifacts/custom/RotatE_mixed_50_50/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **408**
  - `/m/08mbj5d` (score=-8.062)
  - `/m/0n7q7` (score=-8.106)
  - `/m/0fw2y` (score=-8.144)
  - `/m/030qb3t` (score=-8.410)
  - `/m/0xc9x` (score=-8.446)
  - `/m/01n7q` (score=-8.460)
  - `/m/029qzx` (score=-8.481)
  - `/m/01jzyx` (score=-8.484)
  - `/m/02dtg` (score=-8.495)
  - `/m/0nr_q` (score=-8.512)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **5**
  - `/m/01n7q` (score=-8.905)
  - `/m/07ssc` (score=-8.937)
  - `/m/08mbj5d` (score=-9.052)
  - `/m/02jx1` (score=-9.072)
  - `/m/07b_l` (score=-9.172) ←gold
  - `/m/059rby` (score=-9.201)
  - `/m/0345h` (score=-9.263)
  - `/m/0d060g` (score=-9.290)
  - `/m/03rjj` (score=-9.295)
  - `/m/04n6k` (score=-9.310)

### artifacts/custom/RotatE_mixed_70_30/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **320**
  - `/m/08mbj5d` (score=-8.041)
  - `/m/0n7q7` (score=-8.167)
  - `/m/030qb3t` (score=-8.230)
  - `/m/0fw2y` (score=-8.232)
  - `/m/02dtg` (score=-8.373)
  - `/m/02_286` (score=-8.380)
  - `/m/0dyl9` (score=-8.453)
  - `/m/0xc9x` (score=-8.455)
  - `/m/013f9v` (score=-8.469)
  - `/m/0nr_q` (score=-8.469)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **4**
  - `/m/01n7q` (score=-8.958)
  - `/m/07ssc` (score=-9.022)
  - `/m/08mbj5d` (score=-9.023)
  - `/m/07b_l` (score=-9.063) ←gold
  - `/m/059rby` (score=-9.116)
  - `/m/04n6k` (score=-9.145)
  - `/m/02jx1` (score=-9.176)
  - `/m/03rjj` (score=-9.261)
  - `/m/0d060g` (score=-9.337)
  - `/m/0f8l9c` (score=-9.344)

## Triple 4: (/m/01bh6y, /people/person/profession, /m/02hrh1q)

### artifacts/custom/RotatE_random/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-6.767) ←gold
  - `/m/01d_h8` (score=-7.012)
  - `/m/0dxtg` (score=-7.076)
  - `/m/02jknp` (score=-7.077)
  - `/m/03gjzk` (score=-7.109)
  - `/m/0np9r` (score=-7.244)
  - `/m/0cbd2` (score=-7.300)
  - `/m/018gz8` (score=-7.322)
  - `/m/02krf9` (score=-7.331)
  - `/m/0kyk` (score=-7.414)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **27**
  - `/m/08mbj5d` (score=-5.180)
  - `/m/04n6k` (score=-5.935)
  - `/m/0jbk9` (score=-6.093)
  - `/m/04ztj` (score=-6.116)
  - `/m/09nqf` (score=-6.413)
  - `/m/086k8` (score=-6.429)
  - `/m/05zppz` (score=-6.488)
  - `/m/02mjmr` (score=-6.498)
  - `/m/02hcv8` (score=-6.502)
  - `/m/0146pg` (score=-6.546)

### artifacts/custom/RotatE_hard/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.062) ←gold
  - `/m/01d_h8` (score=-7.388)
  - `/m/0dxtg` (score=-7.432)
  - `/m/02jknp` (score=-7.588)
  - `/m/03gjzk` (score=-7.595)
  - `/m/0np9r` (score=-7.748)
  - `/m/018gz8` (score=-7.848)
  - `/m/02krf9` (score=-7.954)
  - `/m/0cbd2` (score=-8.013)
  - `/m/09jwl` (score=-8.176)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **11**
  - `/m/08mbj5d` (score=-6.439)
  - `/m/02mjmr` (score=-6.905)
  - `/m/0157m` (score=-6.910)
  - `/m/045cq` (score=-6.959)
  - `/m/0168ql` (score=-6.979)
  - `/m/08f3b1` (score=-6.988)
  - `/m/01pfkw` (score=-7.030)
  - `/m/016szr` (score=-7.042)
  - `/m/081nh` (score=-7.044)
  - `/m/09b6zr` (score=-7.045)

### artifacts/custom/RotatE_mixed_30_70/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.020) ←gold
  - `/m/01d_h8` (score=-7.454)
  - `/m/0dxtg` (score=-7.520)
  - `/m/02jknp` (score=-7.690)
  - `/m/03gjzk` (score=-7.755)
  - `/m/0np9r` (score=-7.893)
  - `/m/018gz8` (score=-7.914)
  - `/m/0cbd2` (score=-7.991)
  - `/m/02krf9` (score=-8.014)
  - `/m/0kyk` (score=-8.018)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **8**
  - `/m/08mbj5d` (score=-6.577)
  - `/m/081nh` (score=-6.918)
  - `/m/0157m` (score=-6.925)
  - `/m/02mjmr` (score=-6.930)
  - `/m/0d05fv` (score=-6.947)
  - `/m/09b6zr` (score=-6.985)
  - `/m/08f3b1` (score=-7.017)
  - `/m/01bh6y` (score=-7.020) ←gold
  - `/m/03llf8` (score=-7.030)
  - `/m/056wb` (score=-7.033)

### artifacts/custom/RotatE_mixed_50_50/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.130) ←gold
  - `/m/01d_h8` (score=-7.620)
  - `/m/0dxtg` (score=-7.635)
  - `/m/02jknp` (score=-7.841)
  - `/m/03gjzk` (score=-7.897)
  - `/m/0np9r` (score=-7.984)
  - `/m/018gz8` (score=-7.999)
  - `/m/0cbd2` (score=-8.103)
  - `/m/0kyk` (score=-8.134)
  - `/m/02krf9` (score=-8.175)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **19**
  - `/m/08mbj5d` (score=-6.570)
  - `/m/02mjmr` (score=-6.847)
  - `/m/08f3b1` (score=-6.982)
  - `/m/081nh` (score=-7.017)
  - `/m/016szr` (score=-7.026)
  - `/m/0157m` (score=-7.042)
  - `/m/021r7r` (score=-7.068)
  - `/m/09b6zr` (score=-7.081)
  - `/m/0c5vh` (score=-7.085)
  - `/m/045cq` (score=-7.092)

### artifacts/custom/RotatE_mixed_70_30/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.103) ←gold
  - `/m/01d_h8` (score=-7.528)
  - `/m/0dxtg` (score=-7.602)
  - `/m/02jknp` (score=-7.757)
  - `/m/03gjzk` (score=-7.808)
  - `/m/018gz8` (score=-7.962)
  - `/m/0np9r` (score=-8.047)
  - `/m/0cbd2` (score=-8.058)
  - `/m/02krf9` (score=-8.070)
  - `/m/0kyk` (score=-8.213)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **13**
  - `/m/08mbj5d` (score=-6.547)
  - `/m/0168ql` (score=-6.958)
  - `/m/02mjmr` (score=-7.030)
  - `/m/021r7r` (score=-7.039)
  - `/m/016szr` (score=-7.049)
  - `/m/08f3b1` (score=-7.050)
  - `/m/09b6zr` (score=-7.057)
  - `/m/01gg59` (score=-7.065)
  - `/m/0157m` (score=-7.079)
  - `/m/04n7njg` (score=-7.084)

## Triple 5: (/m/03q95r, /people/person/profession, /m/02hrh1q)

### artifacts/custom/RotatE_random/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.344) ←gold
  - `/m/02krf9` (score=-7.676)
  - `/m/0np9r` (score=-7.751)
  - `/m/01d_h8` (score=-7.820)
  - `/m/03gjzk` (score=-7.840)
  - `/m/02jknp` (score=-7.903)
  - `/m/018gz8` (score=-7.992)
  - `/m/0dxtg` (score=-8.069)
  - `/m/0cbd2` (score=-8.085)
  - `/m/0kyk` (score=-8.105)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **320**
  - `/m/08mbj5d` (score=-5.180)
  - `/m/04n6k` (score=-5.935)
  - `/m/0jbk9` (score=-6.093)
  - `/m/04ztj` (score=-6.116)
  - `/m/09nqf` (score=-6.413)
  - `/m/086k8` (score=-6.429)
  - `/m/05zppz` (score=-6.488)
  - `/m/02mjmr` (score=-6.498)
  - `/m/02hcv8` (score=-6.502)
  - `/m/0146pg` (score=-6.546)

### artifacts/custom/RotatE_hard/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.668) ←gold
  - `/m/02krf9` (score=-8.307)
  - `/m/01d_h8` (score=-8.346)
  - `/m/03gjzk` (score=-8.388)
  - `/m/0np9r` (score=-8.462)
  - `/m/02jknp` (score=-8.519)
  - `/m/0dxtg` (score=-8.584)
  - `/m/018gz8` (score=-8.710)
  - `/m/0cbd2` (score=-8.782)
  - `/m/0kyk` (score=-8.874)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **453**
  - `/m/08mbj5d` (score=-6.439)
  - `/m/02mjmr` (score=-6.905)
  - `/m/0157m` (score=-6.910)
  - `/m/045cq` (score=-6.959)
  - `/m/0168ql` (score=-6.979)
  - `/m/08f3b1` (score=-6.988)
  - `/m/01pfkw` (score=-7.030)
  - `/m/016szr` (score=-7.042)
  - `/m/081nh` (score=-7.044)
  - `/m/09b6zr` (score=-7.045)

### artifacts/custom/RotatE_mixed_30_70/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.591) ←gold
  - `/m/02krf9` (score=-8.166)
  - `/m/01d_h8` (score=-8.223)
  - `/m/03gjzk` (score=-8.348)
  - `/m/02jknp` (score=-8.388)
  - `/m/0np9r` (score=-8.404)
  - `/m/0dxtg` (score=-8.417)
  - `/m/018gz8` (score=-8.604)
  - `/m/0cbd2` (score=-8.639)
  - `/m/0kyk` (score=-8.640)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **481**
  - `/m/08mbj5d` (score=-6.577)
  - `/m/081nh` (score=-6.918)
  - `/m/0157m` (score=-6.925)
  - `/m/02mjmr` (score=-6.930)
  - `/m/0d05fv` (score=-6.947)
  - `/m/09b6zr` (score=-6.985)
  - `/m/08f3b1` (score=-7.017)
  - `/m/03llf8` (score=-7.030)
  - `/m/056wb` (score=-7.033)
  - `/m/016szr` (score=-7.057)

### artifacts/custom/RotatE_mixed_50_50/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.496) ←gold
  - `/m/01d_h8` (score=-8.166)
  - `/m/02krf9` (score=-8.196)
  - `/m/0np9r` (score=-8.217)
  - `/m/03gjzk` (score=-8.287)
  - `/m/02jknp` (score=-8.382)
  - `/m/0dxtg` (score=-8.394)
  - `/m/018gz8` (score=-8.441)
  - `/m/0cbd2` (score=-8.560)
  - `/m/0kyk` (score=-8.574)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **268**
  - `/m/08mbj5d` (score=-6.570)
  - `/m/02mjmr` (score=-6.847)
  - `/m/08f3b1` (score=-6.982)
  - `/m/081nh` (score=-7.017)
  - `/m/016szr` (score=-7.026)
  - `/m/0157m` (score=-7.042)
  - `/m/021r7r` (score=-7.068)
  - `/m/09b6zr` (score=-7.081)
  - `/m/0c5vh` (score=-7.085)
  - `/m/045cq` (score=-7.092)

### artifacts/custom/RotatE_mixed_70_30/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.601) ←gold
  - `/m/02krf9` (score=-8.257)
  - `/m/01d_h8` (score=-8.308)
  - `/m/03gjzk` (score=-8.382)
  - `/m/0np9r` (score=-8.382)
  - `/m/02jknp` (score=-8.490)
  - `/m/0dxtg` (score=-8.510)
  - `/m/018gz8` (score=-8.599)
  - `/m/0cbd2` (score=-8.637)
  - `/m/0kyk` (score=-8.679)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **404**
  - `/m/08mbj5d` (score=-6.547)
  - `/m/0168ql` (score=-6.958)
  - `/m/02mjmr` (score=-7.030)
  - `/m/021r7r` (score=-7.039)
  - `/m/016szr` (score=-7.049)
  - `/m/08f3b1` (score=-7.050)
  - `/m/09b6zr` (score=-7.057)
  - `/m/01gg59` (score=-7.065)
  - `/m/0157m` (score=-7.079)
  - `/m/04n7njg` (score=-7.084)

