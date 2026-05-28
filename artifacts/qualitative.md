# Qualitative predictions on FB15k-237

_Sampled 5 test triples (seed=42). Top-10 candidates per side, filtered against the train+val+test triples._

## Triple 1: (/m/0btyf5z, /film/film/release_date_s./film/film_regional_release_date/film_release_region, /m/0345h)

### artifacts/baseline/pykeen_TransE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **2**
  - `/m/09c7w0` (score=-6.095)
  - `/m/0345h` (score=-6.393) ←gold
  - `/m/0chghy` (score=-6.685)
  - `/m/01mjq` (score=-6.828)
  - `/m/03rt9` (score=-6.839)
  - `/m/03_3d` (score=-6.954)
  - `/m/0154j` (score=-7.042)
  - `/m/035qy` (score=-7.044)
  - `/m/06mkj` (score=-7.050)
  - `/m/05v8c` (score=-7.050)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **31**
  - `/m/03yvf2` (score=-5.305)
  - `/m/0mbql` (score=-5.373)
  - `/m/0gvvm6l` (score=-5.407)
  - `/m/04yg13l` (score=-5.433)
  - `/m/032clf` (score=-5.452)
  - `/m/0m3gy` (score=-5.579)
  - `/m/0bs8hvm` (score=-5.643)
  - `/m/0184tc` (score=-5.682)
  - `/m/0m491` (score=-5.723)
  - `/m/0h21v2` (score=-5.801)

### artifacts/baseline/pykeen_RotatE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/0345h` = **3**
  - `/m/0jgd` (score=-6.800)
  - `/m/06mkj` (score=-6.823)
  - `/m/0345h` (score=-6.893) ←gold
  - `/m/0154j` (score=-6.933)
  - `/m/015fr` (score=-6.941)
  - `/m/035qy` (score=-6.999)
  - `/m/0chghy` (score=-7.048)
  - `/m/03_3d` (score=-7.069)
  - `/m/03spz` (score=-7.070)
  - `/m/05qhw` (score=-7.121)
- **Head prediction**: filtered rank of gold head `/m/0btyf5z` = **48**
  - `/m/049w1q` (score=-5.873)
  - `/m/03yvf2` (score=-5.927)
  - `/m/09gmmt6` (score=-5.975)
  - `/m/0gh65c5` (score=-6.205)
  - `/m/0gvvm6l` (score=-6.301)
  - `/m/0gvvf4j` (score=-6.302)
  - `/m/09146g` (score=-6.442)
  - `/m/01shy7` (score=-6.449)
  - `/m/0gfh84d` (score=-6.450)
  - `/m/0432_5` (score=-6.521)

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

## Triple 2: (/m/03ryks, /music/artist/track_contributions./music/track_contribution/role, /m/013y1f)

### artifacts/baseline/pykeen_TransE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **1**
  - `/m/013y1f` (score=-5.569) ←gold
  - `/m/026t6` (score=-5.585)
  - `/m/05148p4` (score=-5.722)
  - `/m/0342h` (score=-5.887)
  - `/m/042v_gx` (score=-6.047)
  - `/m/07brj` (score=-6.090)
  - `/m/01vj9c` (score=-6.105)
  - `/m/03gvt` (score=-6.355)
  - `/m/07y_7` (score=-6.365)
  - `/m/01v1d8` (score=-6.375)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **2**
  - `/m/032t2z` (score=-5.498)
  - `/m/03ryks` (score=-5.569) ←gold
  - `/m/0f0qfz` (score=-5.575)
  - `/m/01vsksr` (score=-5.575)
  - `/m/023l9y` (score=-5.692)
  - `/m/01vrnsk` (score=-5.770)
  - `/m/01304j` (score=-5.791)
  - `/m/01tp5bj` (score=-5.985)
  - `/m/043c4j` (score=-6.076)
  - `/m/01vng3b` (score=-6.105)

### artifacts/baseline/pykeen_RotatE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013y1f` = **2**
  - `/m/026t6` (score=-5.892)
  - `/m/013y1f` (score=-6.060) ←gold
  - `/m/07brj` (score=-6.088)
  - `/m/0342h` (score=-6.108)
  - `/m/05148p4` (score=-6.123)
  - `/m/042v_gx` (score=-6.156)
  - `/m/01vj9c` (score=-6.177)
  - `/m/07y_7` (score=-6.235)
  - `/m/03qjg` (score=-6.299)
  - `/m/01v1d8` (score=-6.426)
- **Head prediction**: filtered rank of gold head `/m/03ryks` = **13**
  - `/m/0140t7` (score=-5.669)
  - `/m/082brv` (score=-5.698)
  - `/m/01l4g5` (score=-5.886)
  - `/m/01mwsnc` (score=-5.913)
  - `/m/01nn3m` (score=-5.926)
  - `/m/07_3qd` (score=-5.934)
  - `/m/023slg` (score=-5.991)
  - `/m/01w923` (score=-6.011)
  - `/m/01lvcs1` (score=-6.019)
  - `/m/01vsksr` (score=-6.025)

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

## Triple 3: (/m/07b_l, /location/location/contains, /m/013m4v)

### artifacts/baseline/pykeen_TransE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **168**
  - `/m/0vzm` (score=-6.376)
  - `/m/07b_l` (score=-6.402)
  - `/m/03l2n` (score=-6.901)
  - `/m/0d35y` (score=-7.056)
  - `/m/0tbql` (score=-7.236)
  - `/m/0fr0t` (score=-7.281)
  - `/m/05jbn` (score=-7.446)
  - `/m/02j3w` (score=-7.449)
  - `/m/0djd3` (score=-7.489)
  - `/m/01pl14` (score=-7.532)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **6**
  - `/m/013m4v` (score=-6.402)
  - `/m/030qb3t` (score=-8.217)
  - `/m/059rby` (score=-8.397)
  - `/m/0d060g` (score=-8.410)
  - `/m/02_286` (score=-8.411)
  - `/m/07b_l` (score=-8.501) ←gold
  - `/m/0f2rq` (score=-8.505)
  - `/m/01n7q` (score=-8.559)
  - `/m/05fjf` (score=-8.692)
  - `/m/0mr_8` (score=-8.708)

### artifacts/baseline/pykeen_RotatE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/013m4v` = **1138**
  - `/m/08mbj5d` (score=-7.256)
  - `/m/04n6k` (score=-7.504)
  - `/m/04ztj` (score=-7.793)
  - `/m/02hcv8` (score=-7.855)
  - `/m/06jk5_` (score=-7.866)
  - `/m/030qb3t` (score=-7.926)
  - `/m/0cv3w` (score=-7.964)
  - `/m/05zppz` (score=-7.969)
  - `/m/0vzm` (score=-7.980)
  - `/m/05njyy` (score=-7.994)
- **Head prediction**: filtered rank of gold head `/m/07b_l` = **14**
  - `/m/08mbj5d` (score=-7.984)
  - `/m/02jx1` (score=-8.624)
  - `/m/0jbk9` (score=-8.627)
  - `/m/07ssc` (score=-8.671)
  - `/m/05zppz` (score=-8.817)
  - `/m/01n7q` (score=-8.858)
  - `/m/0d060g` (score=-8.918)
  - `/m/03rk0` (score=-9.056)
  - `/m/02hcv8` (score=-9.076)
  - `/m/0f8l9c` (score=-9.096)

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

## Triple 4: (/m/01bh6y, /people/person/profession, /m/02hrh1q)

### artifacts/baseline/pykeen_TransE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-6.111) ←gold
  - `/m/01d_h8` (score=-6.846)
  - `/m/02jknp` (score=-6.866)
  - `/m/0np9r` (score=-6.880)
  - `/m/0dxtg` (score=-7.048)
  - `/m/02krf9` (score=-7.134)
  - `/m/018gz8` (score=-7.192)
  - `/m/0d1pc` (score=-7.492)
  - `/m/03gjzk` (score=-7.498)
  - `/m/015cjr` (score=-7.574)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **1**
  - `/m/01bh6y` (score=-6.111) ←gold
  - `/m/02mjmr` (score=-6.275)
  - `/m/0157m` (score=-6.429)
  - `/m/06y3r` (score=-6.610)
  - `/m/01htxr` (score=-6.704)
  - `/m/0chrwb` (score=-6.718)
  - `/m/01pw9v` (score=-6.725)
  - `/m/07g2v` (score=-6.741)
  - `/m/08p1gp` (score=-6.755)
  - `/m/0c5vh` (score=-6.773)

### artifacts/baseline/pykeen_RotatE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-6.703) ←gold
  - `/m/01d_h8` (score=-7.167)
  - `/m/0np9r` (score=-7.188)
  - `/m/0dxtg` (score=-7.236)
  - `/m/02jknp` (score=-7.244)
  - `/m/03gjzk` (score=-7.461)
  - `/m/018gz8` (score=-7.500)
  - `/m/02krf9` (score=-7.589)
  - `/m/0kyk` (score=-7.593)
  - `/m/0cbd2` (score=-7.629)
- **Head prediction**: filtered rank of gold head `/m/01bh6y` = **10**
  - `/m/08mbj5d` (score=-5.573)
  - `/m/02mjmr` (score=-6.321)
  - `/m/05zppz` (score=-6.363)
  - `/m/04ztj` (score=-6.549)
  - `/m/0157m` (score=-6.555)
  - `/m/016szr` (score=-6.603)
  - `/m/02779r4` (score=-6.667)
  - `/m/01gg59` (score=-6.676)
  - `/m/04n6k` (score=-6.680)
  - `/m/01bh6y` (score=-6.703) ←gold

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

## Triple 5: (/m/03q95r, /people/person/profession, /m/02hrh1q)

### artifacts/baseline/pykeen_TransE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.232) ←gold
  - `/m/02krf9` (score=-7.508)
  - `/m/0np9r` (score=-7.558)
  - `/m/0dxtg` (score=-7.822)
  - `/m/01d_h8` (score=-7.858)
  - `/m/018gz8` (score=-7.895)
  - `/m/03gjzk` (score=-7.938)
  - `/m/02jknp` (score=-7.948)
  - `/m/021wpb` (score=-8.326)
  - `/m/0cbd2` (score=-8.339)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **130**
  - `/m/02mjmr` (score=-6.275)
  - `/m/0157m` (score=-6.429)
  - `/m/06y3r` (score=-6.610)
  - `/m/01htxr` (score=-6.704)
  - `/m/0chrwb` (score=-6.718)
  - `/m/01pw9v` (score=-6.725)
  - `/m/07g2v` (score=-6.741)
  - `/m/08p1gp` (score=-6.755)
  - `/m/0c5vh` (score=-6.773)
  - `/m/04ns3gy` (score=-6.774)

### artifacts/baseline/pykeen_RotatE/trained_model.pkl
- **Tail prediction**: filtered rank of gold tail `/m/02hrh1q` = **1**
  - `/m/02hrh1q` (score=-7.318) ←gold
  - `/m/02krf9` (score=-7.726)
  - `/m/03gjzk` (score=-7.761)
  - `/m/01d_h8` (score=-7.762)
  - `/m/018gz8` (score=-7.767)
  - `/m/0np9r` (score=-7.792)
  - `/m/02jknp` (score=-7.827)
  - `/m/0dxtg` (score=-7.897)
  - `/m/0d1pc` (score=-8.371)
  - `/m/0kyk` (score=-8.487)
- **Head prediction**: filtered rank of gold head `/m/03q95r` = **154**
  - `/m/08mbj5d` (score=-5.573)
  - `/m/02mjmr` (score=-6.321)
  - `/m/05zppz` (score=-6.363)
  - `/m/04ztj` (score=-6.549)
  - `/m/0157m` (score=-6.555)
  - `/m/016szr` (score=-6.603)
  - `/m/02779r4` (score=-6.667)
  - `/m/01gg59` (score=-6.676)
  - `/m/04n6k` (score=-6.680)
  - `/m/09b6zr` (score=-6.715)

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

