import matplotlib.pyplot as plt

hidden_units = [5, 20, 50, 100, 200]
train_loss = [2.198164730065944, 1.995322969147532, 1.8505282452013148, 1.6247354451615086, 1.3800203541540699]  # Replace with your last epoch train loss data
validation_loss = [2.2046939640152425, 2.019795048013703, 1.89666168348911, 1.7225381492369667, 1.5431596732282367]  # Replace with your last epoch validation loss data

plt.figure(figsize=(10, 6))
plt.plot(hidden_units, train_loss, marker='o', label='Training Loss')
plt.plot(hidden_units, validation_loss, marker='s', label='Validation Loss')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Average Cross-Entropy Loss')
plt.title('Average Train and Validation Cross-Entropy Loss vs Number of Hidden Units')
plt.legend()
plt.grid(True)
plt.show()

epochs = list(range(1, 101))

train_loss_lr03 = [2.1830256041750356, 1.7664774337395996, 1.4899790294042592, 1.199982248413709, 0.9602423273161699, 0.8017103725527275, 0.7030217737172414, 0.6410041446688644, 0.5529790553368177, 0.49592181262558355, 0.4508839423854112, 0.41113294762831826, 0.3802514665788185, 0.345514479617487, 0.32725116438548524, 0.3004473453714522, 0.29207812063976735, 0.272326682820238, 0.2466350662652302, 0.22943166369038573, 0.21882758081272838, 0.2168987135924819, 0.18925865893933422, 0.17936024711393084, 0.17507537286156197, 0.1618106584476372, 0.15915899401472428, 0.1409246995219166, 0.1319975860889317, 0.12209221410587894, 0.1157733384567109, 0.11294571391084969, 0.10299503034634941, 0.0984261141945659, 0.09136715965820673, 0.09071418662350655, 0.08247650826351105, 0.07868615117784228, 0.07457408202026888, 0.08702161654317087, 0.06690516142787878, 0.06489885856712739, 0.06251711013009341, 0.059933523955133905, 0.06392914482736209, 0.05461417555195899, 0.055985630541748564, 0.05129139443620375, 0.04879252594058295, 0.04906313264992248, 0.04679258929010008, 0.0447774451814876, 0.04298378826506443, 0.04198047554506522, 0.04383998511682478, 0.041822960241143556, 0.03878943448221691, 0.040171698696740366, 0.03658357607015749, 0.03540918039106002, 0.034894278178013056, 0.0347482845510796, 0.03317297830830001, 0.03287232044556238, 0.031945805327374385, 0.03210040674929403, 0.03072254913566058, 0.029598106950262923, 0.029109873273145927, 0.03209844804888785, 0.02843712018020024, 0.028227960577567036, 0.028987567393855482, 0.026320351362153237, 0.025749404555512567, 0.026058389029316297, 0.025410988006921962, 0.02457700767282456, 0.024481692486928507, 0.025217956344479736, 0.026583888042715054, 0.02362012316029355, 0.025346247880167348, 0.023106544044351952, 0.022847138668457737, 0.023057045624028813, 0.02253767687827087, 0.0219408813962408, 0.022733725956167076, 0.021213105442457546, 0.02131836813829904, 0.020491049818298884, 0.020188011112373942, 0.019812268903914496, 0.020067850766250034, 0.022816345450875266, 0.020848448856199953, 0.019763829471650386, 0.019052778038674726, 0.02074239904461666]
validation_loss_lr03 = [2.2037438353109287, 1.8206886782092007, 1.6045149857067043, 1.4004750971504607, 1.2224731634759607, 1.1520499776776196, 1.0980062589168655, 1.1054995434404393, 1.084246326090598, 1.0885218512841226, 1.0818858861896479, 1.0999836702835983, 1.1172369599184924, 1.1045848469624884, 1.1585278206737712, 1.139799547086517, 1.1492708090567085, 1.218395881726755, 1.1853362931045799, 1.1805676835265344, 1.2263774852218525, 1.3002338227900347, 1.2454073009752529, 1.2999597234119513, 1.2891344607460544, 1.3525253715024492, 1.3026146505439344, 1.3764095507219312, 1.3746419886949297, 1.3777516320903092, 1.4021456306904425, 1.423579161354083, 1.4228707376379848, 1.453429857139848, 1.4440534652833876, 1.4449227137917169, 1.4773992193106293, 1.4752791219913843, 1.5181566642332385, 1.495147549163259, 1.5355384393368028, 1.5382669012165298, 1.5414303114673982, 1.5808234730010287, 1.5322056278422531, 1.5869163834971607, 1.6391242634975318, 1.6214981412973535, 1.60484540953458, 1.636803817358707, 1.6176029280497337, 1.6547595815528178, 1.646783850297035, 1.6725596429615908, 1.711484471392629, 1.7056252271042263, 1.6920253792851383, 1.7181311980754073, 1.7012890609445, 1.7050789274922173, 1.7183259475409736, 1.7131322898479855, 1.7317627778199594, 1.7168942866037518, 1.743477098637676, 1.7275408889313484, 1.7422679322167178, 1.7517846938474044, 1.7495846181151489, 1.7340157385315258, 1.7842198597447976, 1.8004597963421956, 1.7561184711997992, 1.7838419551354503, 1.7914071661618634, 1.8094466466739032, 1.813709188904934, 1.801918150454106, 1.8243717438252434, 1.8417362360852871, 1.8673453856352942, 1.8404836155254827, 1.874500532151967, 1.8565186968367835, 1.8259481380939846, 1.8266331086713288, 1.8805784537018968, 1.8684979684233536, 1.8328850049839165, 1.8816265199679696, 1.8896086622149144, 1.8585471327330276, 1.8663461024900938, 1.8891474031397062, 1.8702874704205557, 1.865052763876409, 1.8793941898218165, 1.9213565817444929, 1.9100114572391467, 1.8779847260255846]

train_loss_lr003 = [2.3008417282746008, 2.288852532895633, 2.278440159699887, 2.2649790344000094, 2.2497283903073195, 2.232544691592468, 2.2142386721316516, 2.1889797521582253, 2.15984818276916, 2.127092792920885, 2.0910989839204817, 2.052569351808884, 2.0099266318806697, 1.9658181567489477, 1.9236031478109248, 1.8805849514748352, 1.8395867823610035, 1.798489114045382, 1.7628515187996672, 1.722016940935073, 1.6871464665109963, 1.650714192096891, 1.61651507558257, 1.5832522399337174, 1.5539156521863005, 1.5215836679475108, 1.4873353986271907, 1.4567147075688964, 1.42538874653655, 1.3950571372484606, 1.364122852668603, 1.3345099940177219, 1.3058418528063565, 1.2766024832335456, 1.2498894004300913, 1.2222487102190471, 1.1949291460456826, 1.168781900367493, 1.1441465475008616, 1.117119917891442, 1.0937315150338713, 1.0703553161090607, 1.047775465167071, 1.024476104740158, 1.00442904671918, 0.9824260910801962, 0.9626010207084988, 0.9432773229300921, 0.9243473810198761, 0.9061473754428846, 0.8897806588570311, 0.8725851218723173, 0.856647933017519, 0.840782188931733, 0.8255238578099902, 0.8119187856449334, 0.796760005621542, 0.7841724215942819, 0.7705073198245771, 0.7583547201242073, 0.7458210468579105, 0.7342776355430941, 0.7228845066425426, 0.7118263799125248, 0.701715825888597, 0.6909955393718412, 0.6811542883192515, 0.6717855768579801, 0.6629270847234954, 0.6539596610427036, 0.6449164416674528, 0.6363103534872168, 0.6285625043232915, 0.6200204145927697, 0.6126311311441546, 0.6053032151181644, 0.5973060864584981, 0.5900347007448669, 0.5828979748337565, 0.5763943216695743, 0.5701063253273779, 0.5630158103082257, 0.5567166787173344, 0.5504845469003455, 0.5444050458808252, 0.5384214833069253, 0.5330339405363173, 0.5270876904506384, 0.5224962995099613, 0.516301999952403, 0.5107495071975997, 0.5057120446567269, 0.5010635715167243, 0.4953450920656594, 0.4907090332008313, 0.48572422443009217, 0.4812126366448187, 0.47621540770000115, 0.4721365493031168, 0.46742089104162104]
validation_loss_lr003 = [2.2998297076737564, 2.2896331759679627, 2.2810219877388476, 2.269682214169886, 2.2565414494900873, 2.2414121067935575, 2.22514275312563, 2.2026698456676743, 2.176373762991747, 2.1469848847812827, 2.1139062614628337, 2.078971442821732, 2.040384121797004, 1.9999717296027109, 1.9623127625999024, 1.9234083256293055, 1.8871262861830977, 1.8515173716390956, 1.8207469301621793, 1.7856587380491602, 1.756633775168137, 1.7266908560747296, 1.6992425761784045, 1.672408040131662, 1.6486703031787886, 1.6250702521776685, 1.5976077757740075, 1.5734256021828892, 1.5505706573533686, 1.5272543120644841, 1.503836934734129, 1.4825429187581058, 1.4623496528192, 1.4393379047839403, 1.4196916932164243, 1.4006504850841952, 1.3836252275985432, 1.3637292872744846, 1.3453588516254245, 1.328875350292482, 1.3126168216985894, 1.295759637022772, 1.2813580998304164, 1.2661563628832857, 1.2520456389231343, 1.2394008659476983, 1.2285235908030545, 1.217273966439516, 1.203716441752266, 1.1943242191259846, 1.186713540572749, 1.1778910954625648, 1.165807876273377, 1.1597940965492952, 1.1514397482582164, 1.1449889045081807, 1.1337555050679087, 1.1330397106626093, 1.1231449836652931, 1.1208855053479787, 1.1147781571907673, 1.108693429527203, 1.1021733070090043, 1.0994292738390903, 1.0972710741971845, 1.0934427247834957, 1.0905195236478722, 1.0859291131893618, 1.083954366124312, 1.0834494349998216, 1.0811055604434243, 1.0788370113363919, 1.0717491274818358, 1.073631539495599, 1.0715561483942377, 1.0697821832913967, 1.0693993864985072, 1.0680998432507052, 1.0649196995515116, 1.0651949136258576, 1.0683958741360249, 1.0639089949209821, 1.0615606817958507, 1.0589075500842116, 1.0596201995087156, 1.058806635285969, 1.0599503023500148, 1.0584172545387578, 1.061570649873172, 1.059035658961993, 1.060325556625385, 1.057322868483569, 1.0568729609996206, 1.0587326575636777, 1.0587851308943024, 1.0598719003600796, 1.0628913076477182, 1.0608468311922148, 1.0652718343095962, 1.0598508756432812]

train_loss_lr0003 = [2.3142659147349387, 2.3105798696705784, 2.3076565225802423, 2.3055470482426035, 2.3038286477948255, 2.3023101661488585, 2.300996811044057, 2.2998012660530325, 2.2986795325352594, 2.2975834436573503, 2.2965270942767435, 2.2954790732820496, 2.2944339810419008, 2.293401355572182, 2.292367522082581, 2.291340057016787, 2.290302498183907, 2.2892552058532134, 2.2881999386995657, 2.287137720445218, 2.2860677614831344, 2.284986183449723, 2.2838952267853316, 2.2827947696089557, 2.2816824267974565, 2.280559575913382, 2.2794219154321227, 2.2782700911499543, 2.2771027059901527, 2.2759257711437035, 2.274730423262271, 2.2735204084019003, 2.272294628838777, 2.2710539154409344, 2.2697957437384053, 2.2685204441344466, 2.267225714269323, 2.2659142775641095, 2.2645845378621465, 2.263232614514236, 2.2618606631370644, 2.260470618724501, 2.259055945442032, 2.2576201802232747, 2.256159697279048, 2.254678215944829, 2.253171251531268, 2.2516413651869973, 2.250086494179631, 2.2485072139360227, 2.246900814062177, 2.245265447082168, 2.243604757901772, 2.2419159883577855, 2.240197753289679, 2.2384519305501343, 2.236675690477401, 2.2348698638473032, 2.233031352689246, 2.231163083303654, 2.2292598065332028, 2.227326599355034, 2.225359564430766, 2.223359240786904, 2.22132525818372, 2.2192562590551934, 2.2171519151324466, 2.21501272822673, 2.2128379263912707, 2.210625990806273, 2.208377076247317, 2.2060883266727154, 2.203762084376248, 2.201398517756974, 2.1989973909398812, 2.1965562075809113, 2.194074794913249, 2.1915543761781127, 2.188992221888617, 2.1863905808665813, 2.18374846067942, 2.1810667243585677, 2.178340859318907, 2.1755753212297675, 2.1727677355826627, 2.169920806297663, 2.167031687791875, 2.1641011427270893, 2.1611286781762153, 2.158114686417946, 2.1550564143603888, 2.151956509341596, 2.148821645000744, 2.145640733561929, 2.1424183094623954, 2.139155655369871, 2.1358510410086597, 2.132506813752935, 2.1291220809275884, 2.125698178797254]
validation_loss_lr003 = [2.3117092301310027, 2.308187172591561, 2.305428689745677, 2.3034907783473484, 2.3019463778184415, 2.3005984567050985, 2.2994539553392745, 2.2984394480153187, 2.297498172851219, 2.29658334933453, 2.295705920461489, 2.2948454700969436, 2.2939818447577864, 2.293133884231989, 2.2922882056226626, 2.2914444147579944, 2.29058800430536, 2.289725162039293, 2.2888497224816224, 2.2879714767793256, 2.287089994710425, 2.2861948363123425, 2.285293052838338, 2.284379894321588, 2.283461021127542, 2.2825251735893066, 2.281574240429977, 2.2806056325222137, 2.279629548830902, 2.2786387381666438, 2.277634412660226, 2.276620411403099, 2.275592718915256, 2.2745441994039637, 2.273476510246169, 2.272399777602977, 2.2713045021046954, 2.2701912240025166, 2.269061908680305, 2.2679110818053716, 2.2667415543963134, 2.2655497706441268, 2.2643450324837118, 2.263114645601586, 2.26185339446277, 2.260575797261843, 2.259274196398919, 2.257956336689876, 2.256610782118364, 2.2552426820406914, 2.2538475175447967, 2.2524305146103814, 2.2509818545844995, 2.249511124286961, 2.2480104403020333, 2.246486527365696, 2.2449314234206073, 2.2433469820956926, 2.241735011558214, 2.2400904398023957, 2.2384152497737215, 2.236712467981919, 2.2349781697343496, 2.2332102462451475, 2.2314133014237494, 2.229585500458496, 2.2277248146634756, 2.225825116655794, 2.223895896073377, 2.2219274803232545, 2.219925707945665, 2.217887959451372, 2.2158113374068957, 2.2137030993398095, 2.2115639286192827, 2.2093731343764533, 2.2071531479445596, 2.2048927446182915, 2.2025944375250903, 2.200259490650124, 2.1978886933372026, 2.195480155346937, 2.1930235377946214, 2.190530262492266, 2.187993520053449, 2.185425615430456, 2.182814718933738, 2.1801668025353247, 2.1774746691135305, 2.1747493404932943, 2.17198693292491, 2.169181465659517, 2.1663424414988874, 2.163458407794889, 2.160527754117074, 2.1575680416455096, 2.1545733089592245, 2.151534526875005, 2.148467218583346, 2.145352896343387]

# Plot for learning rate 0.03
plt.figure(figsize=(14, 7))
plt.plot(epochs, train_loss_lr03, label='Training Loss (LR=0.03)')
plt.plot(epochs, validation_loss_lr03, label='Validation Loss (LR=0.03)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch (LR=0.03)')
plt.legend()
plt.grid(True)
plt.show()

# Plot for learning rate 0.003
plt.figure(figsize=(14, 7))
plt.plot(epochs, train_loss_lr003, label='Training Loss (LR=0.003)')
plt.plot(epochs, validation_loss_lr003, label='Validation Loss (LR=0.003)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch (LR=0.003)')
plt.legend()
plt.grid(True)
plt.show()


# Plot for learning rate 0.0003
plt.figure(figsize=(14, 7))
plt.plot(epochs, train_loss_lr0003, label='Training Loss (LR=0.0003)')
plt.plot(epochs, validation_loss_lr003, label='Validation Loss (LR=0.0003)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs. Epoch (LR=0.0003)')
plt.legend()
plt.grid(True)
plt.show()