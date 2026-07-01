#Imports
from math import (pi, fabs, sqrt, log, pow, exp, tan, sin, cos, acos, isnan,
                  atan)

from matrix_operations import (eig_vals_3x3_analytical_erick,
                                matrix_multiply_vector, flat_to_voight,
                               voight_to_flat, matrix_transpose,
                               matrix_multiply, matrix_eigenvectors,
                               matrix_eigenvectors_erick, matrix_zero)

from compyle.api import declare

def pegasus_alg(props = [0], stress0_flat = [0], stVar0 = [0], eps = [0],
               stressN_flat = [0], stVarN = [0], Ftol = [0], maxits = [0],
                pegasus = [0], pegasus0 = [0], pegasus1 = [0], idx=0,
                yield_old = 1.0):
   

    #---------------------------------------------------------------------------
    #Declarando variaveis
    #---------------------------------------------------------------------------
    
    i = declare("int", 1)

    yieldN = declare("float")
    
    DE = declare("matrix(36)") #matriz elástica vetorizada
    DE_0 = declare("matrix(36)") #matriz elástica vetorizada
    DE_1 = declare("matrix(36)") #matriz elástica vetorizada
    
    eigvals0 = declare("matrix(3)") #tensões principais iniciais

    stress_inc = declare("matrix(6)") #incremento de tensões
    stress_inc_flat = declare("matrix(9)") #incremento de tensões
    stressI = declare("matrix(6)") #tensões após previsão elástica
    stressI_flat = declare("matrix(9)") #tensões após previsão elástica
    eigvalsI = declare("matrix(3)") #tensões principais iniciais
    
    stress_inc_0 = declare("matrix(6)") #incremento de tensões
    stress_inc_0_flat = declare("matrix(9)") #incremento de tensões
    stressI_0 = declare("matrix(6)") #tensões após previsão elástica
    stressI_0_flat = declare("matrix(9)") #tensões após previsão elástica
    eigvalsI_0 = declare("matrix(3)") #tensões principais iniciais
    
    stress_inc_1 = declare("matrix(6)") #incremento de tensões
    stress_inc_1_flat = declare("matrix(9)") #incremento de tensões
    stressI_1 = declare("matrix(6)") #tensões após previsão elástica
    stressI_1_flat = declare("matrix(9)") #tensões após previsão elástica
    eigvalsI_1 = declare("matrix(3)") #tensões principais iniciais

    matrix_zero(DE, 36)
    matrix_zero(DE_0, 36)
    matrix_zero(DE_1, 36)


    #---------------------------------------------------------------------------
    #Passando variaveis
    #---------------------------------------------------------------------------

    #Passando variáveis de estado para variáveis locais
    ocr = stVar0[1] #Passando OCR
    e   = stVar0[2] #Passando índice de vazios
    y   = stVar0[3] #Passando o parâmetro de estado
    p0  = stVar0[4] #Passando tensão de pré-adensamento
    v   = stVar0[6] #Passando índice de vazios específico
    

    #Passando propriedades para variaveis locais, Variável
    kappa   =  props[0]  # k
    nu      =  props[1]  # nu
    gamma   =  props[2]  # gamma
    _lambda =  props[3]  # lambda
    n       =  props[4]  # n, não confundir com pN que a tensão média nova
    r       =  props[5]  # r
    m       =  props[6]  # m
    phi     =  props[7]  # ângulo de atrito em graus
    p1      =  props[8]  # tensão de referência para gamma
    ratioPS =  props[9]  # controlador de psmall
    lode    =  props[10] # habilita a variação do ângulo de Lode, se for diferente de zero
    psmall  =  props[18] #tensão mínima a ser considerada
    Mtc     = props [19] #M na condição de compressão triaxial
    a       = props [20] #parâmetro usado na equação de Mteta

    #---------------------------------------------------------------------------
    #Cálculo da superfície de plastificação inicial
    #---------------------------------------------------------------------------
    
    #Usa-se p0 e Mteta0, para deixar a suerfície de plastificação inicial parada e avaliar o erro em relaçã a ela.

    #Calculando tensões principais iniciais
    eig_vals_3x3_analytical_erick(stress0_flat, eigvals0)
    s1_0 = - eigvals0[0] #compressão positivo
    s2_0 = - eigvals0[1] 
    s3_0 = - eigvals0[2] 

    #Calculando os invariantes de cambrigde iniciais
    p_0 = (s1_0 + s2_0 + s3_0) / 3.0    
    q_0 = sqrt(((s1_0 - s2_0)**2.0) + ((s2_0 - s3_0)**2.0) + 
            ((s3_0 - s1_0)**2.0)) / sqrt(2.0)
    teta0 = 0.0
    Mteta0 = 0.0
    if lode == 0.0: 
        teta0 = 30.0 #Mlode = Mtc
        Mteta0 = Mtc
    else:
        if fabs(s1_0  - s3_0) < psmall: 
            teta0 = 30.0 #condição isotrópica
            Mteta0 = Mtc  
        else:
            teta0 = atan((s1_0 - 2.0 * s2_0 + s3_0) / 
                    (sqrt(3.0)*(s1_0 - s3_0))) * 180.0 / pi  
            Mteta0 = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                    (sin(-3.0 * teta0 * pi / 180.0))))**(1.0 / 4.0)
    
    #---------------------------------------------------------------
    #Previsão Elástica 0
    #---------------------------------------------------------------
    
    #Matriz elástica 0
    #evInc_0 = - pegasus0[0] * (eps[0] + eps[1] + eps[2]) #incremento de deformacaovolumetrica, compressão positivo
    #bulk_0 = p_0 * (exp (v * evInc_0 / kappa) - 1.0) / evInc_0 
    bulk_0 = v * p_0 / kappa 
    #if evInc_0 == 0.0: bulk_0 = v * p_0 / kappa #
    if bulk_0 < 10e3 * p1: bulk_0 = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
    if bulk_0 > 100e6 * p1: bulk_0 = 100e6 * p1  #limite superior bulk=100 GPa
    shear_0 = 1.5 * bulk_0 * (1.0 - 2.0 * nu) / (1.0 + nu) 
    alpha1_0 = bulk_0 + 4.0 * shear_0 / 3.0
    alpha2_0 = bulk_0 - 2.0 * shear_0 / 3.0

    DE_0 [0]  = alpha1_0
    DE_0 [7]  = alpha1_0
    DE_0 [14] = alpha1_0
    DE_0 [1]  = alpha2_0
    DE_0 [2]  = alpha2_0
    DE_0 [6]  = alpha2_0
    DE_0 [8]  = alpha2_0
    DE_0 [12] = alpha2_0
    DE_0 [13] = alpha2_0
    DE_0 [21] = 2.0 * shear_0
    DE_0 [28] = 2.0 * shear_0
    DE_0 [35] = 2.0 * shear_0
    
    #Previsao Elastica 0
    matrix_multiply_vector(DE_0, eps, stress_inc_0, 6) #multiplicação stress_inc = DE * eps
    voight_to_flat (stress_inc_0, stress_inc_0_flat)
    for i in range (9):
        #stressI_0_flat[i] = stress0_flat[i] + stress_inc_0_flat[i] * pegasus0[0]
        stressI_0_flat[i] = stress0_flat[i]
    #Verficação de plastificação F0
    eig_vals_3x3_analytical_erick(stressI_0_flat, eigvalsI_0)
    s1_I_0 = - eigvalsI_0[0] #compressão positivo
    s2_I_0 = - eigvalsI_0[1] 
    s3_I_0 = - eigvalsI_0[2] 

    #Calculando os invariantes de cambrigde
    pI_0 = (s1_I_0 + s2_I_0 + s3_I_0) / 3.0    
    qI_0 = sqrt(((s1_I_0 - s2_I_0)**2.0) + ((s2_I_0 - s3_I_0)**2.0) + 
            ((s3_I_0 - s1_I_0)**2.0)) / sqrt(2.0)
 
    #Verficação de plastificação 0
    yieldI_0 = ((qI_0**n) / ((Mteta0 * pI_0)**n) + 
                log(pI_0 / p0) / log(r))

    #---------------------------------------------------------------
    #Previsão Elástica 1
    #---------------------------------------------------------------
    
    #Matriz elástica 1
    #evInc_1 = - pegasus1[0] * (eps[0] + eps[1] + eps[2]) #incremento de deformacaovolumetrica, compressão positivo
    #bulk_1 = p_0 * (exp (v * evInc_1 / kappa) - 1.0) / evInc_1
    bulk_1 = v * p_0 / kappa 
    #if evInc_1 == 0.0: bulk_1 = v * p_0 / kappa #
    if bulk_1 < 10e3 * p1: bulk_1 = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
    if bulk_1 > 100e6 * p1: bulk_1 = 100e6 * p1  #limite superior bulk=100 GPa
    shear_1 = 1.5 * bulk_1 * (1.0 - 2.0 * nu) / (1.0 + nu) 
    alpha1_1 = bulk_1 + 4.0 * shear_1 / 3.0
    alpha2_1 = bulk_1 - 2.0 * shear_1 / 3.0

    DE_1 [0]  = alpha1_1
    DE_1 [7]  = alpha1_1
    DE_1 [14] = alpha1_1
    DE_1 [1]  = alpha2_1
    DE_1 [2]  = alpha2_1
    DE_1 [6]  = alpha2_1
    DE_1 [8]  = alpha2_1
    DE_1 [12] = alpha2_1
    DE_1 [13] = alpha2_1
    DE_1 [21] = 2.0 * shear_1
    DE_1 [28] = 2.0 * shear_1
    DE_1 [35] = 2.0 * shear_1
    
    #Previsao Elastica 1
    matrix_multiply_vector(DE_1, eps, stress_inc_1, 6) #multiplicação stress_inc = DE * eps
    voight_to_flat (stress_inc_1, stress_inc_1_flat)
    for i in range (9):
        stressI_1_flat[i] = stress0_flat[i] + stress_inc_1_flat[i] * pegasus1[0]

    #Verficação de plastificação F1
    eig_vals_3x3_analytical_erick(stressI_1_flat, eigvalsI_1)
    s1_I_1 = - eigvalsI_1[0] #compressão positivo
    s2_I_1 = - eigvalsI_1[1] 
    s3_I_1 = - eigvalsI_1[2] 

    #Calculando os invariantes de cambrigde
    pI_1 = (s1_I_1 + s2_I_1 + s3_I_1) / 3.0    
    qI_1 = sqrt(((s1_I_1 - s2_I_1)**2.0) + ((s2_I_1 - s3_I_1)**2.0) + 
            ((s3_I_1 - s1_I_1)**2.0)) / sqrt(2.0)
 
    #Verficação de plastificação 1
    yieldI_1 = ((qI_1**n) / ((Mteta0 * pI_1)**n) + 
                log(pI_1 / p0) / log(r))


    #---------------------------------------------------------------
    #Iteracoes
    #---------------------------------------------------------------
    c = 0.0 #contador para controle
    c1 = 0.0 #controlador para controle do regula-falsi
    yieldN = 1.0
    while fabs(yieldN) > Ftol[0] and c < maxits[0] and c1 < 2.0:
        c += 1.0

        if fabs(yieldI_0) > Ftol[0] or fabs(yieldI_1) > Ftol[0]:
            pegasus[0] = pegasus1[0] - yieldI_1 * ((pegasus1[0] - pegasus0[0]) /
                        (yieldI_1 - yieldI_0)) 
        '''
        if pegasus[0] > 1.0 or pegasus[0] < 0.0:
            printf("\nPegasus_Print1")
            printf("\nid\n")
            printf("%d\n",idx)
            printf("c\n")
            printf("%.16e\n",c)
            printf("c1\n")
            printf("%.16e\n",c1)
            printf("pegasus[0]\n")
            printf("%.16e\n",pegasus[0])
            printf("pegasus0[0]\n")
            printf("%.16e\n",pegasus0[0])
            printf("pegasus1[0]\n")
            printf("%.16e\n",pegasus1[0])
            printf("yieldN\n")
            printf("%.16e\n",yieldN)
            printf("yieldI_1\n")
            printf("%.16e\n",yieldI_1)
            printf("yieldI_0\n")
            printf("%.16e\n",yieldI_0)
            printf("yield_old\n")
            printf("%.16e\n",yield_old)
        '''
        #Matriz elástica 
        #evInc = - pegasus[0] * (eps[0] + eps[1] + eps[2])
        #bulk = p_0 * (exp (v * evInc / kappa) - 1.0)/ evInc
        bulk = v * p_0 / kappa  
        #if evInc == 0.0: bulk = v * p_0 / kappa 
        if bulk < 10e3 * p1: bulk = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
        if bulk > 100e6 * p1: bulk = 100e6 * p1  #limite superior bulk=100 GPa
        shear = 1.5 * bulk * (1.0 - 2.0 * nu) / (1.0 + nu) 
        alpha1 = bulk + 4.0 * shear / 3.0
        alpha2 = bulk - 2.0 * shear / 3.0

        DE [0]  = alpha1
        DE [7]  = alpha1
        DE [14] = alpha1
        DE [1]  = alpha2
        DE [2]  = alpha2
        DE [6]  = alpha2
        DE [8]  = alpha2
        DE [12] = alpha2
        DE [13] = alpha2
        DE [21] = 2.0 * shear
        DE [28] = 2.0 * shear
        DE [35] = 2.0 * shear
    
        #Previsao Elastica
        matrix_multiply_vector(DE, eps, stress_inc, 6) #multiplicação stress_inc = DE * eps
        voight_to_flat (stress_inc, stress_inc_flat)
        for i in range (9):
            stressI_flat[i] = stress0_flat[i] + stress_inc_flat[i] * pegasus[0]

        #Verficação de plastificação F
        eig_vals_3x3_analytical_erick(stressI_flat, eigvalsI)
        s1_I = - eigvalsI[0] #compressão positivo
        s2_I = - eigvalsI[1] 
        s3_I = - eigvalsI[2] 

        #Calculando os invariantes de cambrigde
        pI = (s1_I + s2_I + s3_I) / 3.0    
        qI = sqrt(((s1_I - s2_I)**2.0) + ((s2_I - s3_I)**2.0) + 
            ((s3_I - s1_I)**2.0)) / sqrt(2.0)
   
        #Verficação de plastificação 
        yieldN = ((qI**n) / ((Mteta0 * pI)**n) + 
                log(pI / p0) / log(r))

        #Verificação da direção de Fnew
        direction = yieldN * yieldI_0

        method = 0 #parece estar errado isso aqui, mas não sei se deixei apenas pra ele nunca mais entrar no regular falsi

        if method == 1:
            #Regula-Falsi
            #aparentemente o regula-falsi não é usado nunca
            if direction < 0.0: #sentido oposto
                pegasus1[0] = pegasus[0]
                yieldI_1 = yieldN
            else:
                pegasus0[0] = pegasus[0]
                yieldI_0 = yieldN
        else:
            #verificacao se um dos testes ja esta sobre a superficie de plastificacao
            if fabs(yieldI_0) < Ftol[0] or fabs(yieldI_1) < Ftol[0]:
                if fabs(yieldI_0) < Ftol[0]:
                    pegasus[0] = pegasus0[0]
                    yieldN = yieldI_0
                if fabs(yieldI_1) < Ftol[0]:
                    pegasus[0] = pegasus1[0]
                    yieldN = yieldI_1
                for i in range (9):
                    stressI_flat[i] = (stress0_flat[i] + stress_inc_flat[i] *
                                    pegasus[0])
            else:
                #Pegasus (Dowel, 1972)
                if direction < 0.0: #sentido oposto
                    pegasus1[0] = pegasus0[0]
                    yieldI_1 = yieldI_0
                else:
                    yieldI_1 = yieldI_1 * yieldI_0 / (yieldI_0 + yieldN)

                pegasus0[0] = pegasus[0]
                yieldI_0 = yieldN
        
        #testa o regula-falsi se não convergir no pegasus
        #Obs: não sei se isso funciona ou se eu apenas esqueci de remover
        if c >= maxits[0] and method != 1:
            c = 0 #isso colocaria o loop em infinito?
            method = 1 #testa o regula-falsi se não convergir no pegasus; mas existe um method = 0 antes do regular falsi, acho que defini que não passe dentro dele
            c1 = c1 + 1.0 #controlador para só passar uma vez no regula-falsi
            if yieldI_0 > yieldI_1:
                aux1 = yieldI_1
                aux2 = pegasus1[0]
                yieldI_1 = yieldI_0
                pegasus1[0] = pegasus0[0]
                yieldI_0 = aux1
                pegasus0[0] = aux2
    

    #---------------------------------------------------------------
    #Atualização de algumas variáveis
    #---------------------------------------------------------------

    #obs: deixei fora do loop do while, inicialmente estava dentro

    #Esse loop provavelmente é desnecessário, posso já passar o StressI pra fora
    for i in range (9):
            stressN_flat[i] = stressI_flat[i]
    
    pN = - (stressN_flat[0] + stressN_flat[4] + stressN_flat[8]) / 3.0

    #Mteta, não deve ser atualizado
    #r_Mteta = r_Mtc * ((2.0D0 * r_a)/(1.0D0 + r_a + (1.0D0 - r_a) *
    # *        (sin(-3.0D0 * r_tetaN * pi / 180.0D0))))**(1.0D0 / 4.0D0)

    evInc = - pegasus[0] * (eps[0] + eps[1] + eps[2])
    ev = stVar0 [5] + evInc #deformação volumétrica acumulada
    v = stVar0 [13] * (1.0 - ev) #referecnial inicial
    e = v - 1.0
    y = v + _lambda * log(pN / p1) - gamma

    stVarN[2] = e
    stVarN[3] = y
    stVarN[5] = ev
    stVarN[6] = v


    stVarN[0] = stVar0[0]
    stVarN[1] = stVar0[1]
   
    stVarN[4] = stVar0[4]

    stVarN[7] = stVar0[7]
    stVarN[8] = stVar0[8]
    stVarN[9] = stVar0[9]
    stVarN[10] = stVar0[10]
    stVarN[11] = stVar0[11]
    stVarN[12] = stVar0[12]
    stVarN[13] = stVar0[13]
    stVarN[14] = stVar0[14]
    stVarN[15] = stVar0[15]
    stVarN[16] = stVar0[16]
    stVarN[17] = stVar0[17]

    for i in range (18):
        if isnan(stVarN[i]):
            printf("\nERRO_Pegausus_stVarN:\n")
            printf("%d\n",i)
            printf("%.16e\n",stVarN[i])
            printf("\nParticula:")
            printf("%d\n",idx)
            stVarN[i] = stVar0[i]
    
    for i in range (9):
        if isnan(stressN_flat[i]):
            printf("\nERRO_Pegasus_stressN_flat:\n")
            printf("%d\n",i)
            printf("%.16e\n",stressN_flat[i])
            printf("\nParticula:")
            printf("%d\n",idx)
            stressN_flat[i] = stress0_flat[i]
    
    if pegasus[0] > 1.0 or pegasus[0] < 0.0:
        pegasus[0] = 0.0
        for i in range (18):
            stVarN[i] = stVar0[i]
        for i in range (9):
            stressN_flat[i] = stress0_flat[i]
        '''
        printf("\nPegasus_Print2")
        printf("\nid\n")
        printf("%d\n",idx)
        printf("c\n")
        printf("%.16e\n",c)
        printf("c1\n")
        printf("%.16e\n",c1)
        printf("yieldN\n")
        printf("%.16e\n",yieldN)
        printf("pegasus[0]\n")
        printf("%.16e\n",pegasus[0])
        printf("pegasus0[0]\n")
        printf("%.16e\n",pegasus0[0])
        printf("pegasus1[0]\n")
        printf("%.16e\n",pegasus1[0])
        '''


def pegasus_unload(props = [0], stress0_flat = [0], stVar0 = [0], eps = [0],
                Ftol = [0], maxits = [0], nsub = [0], pegasus = [0], 
                pegasus0 = [0], pegasus1 = [0], idx=0):
   

    #---------------------------------------------------------------------------
    #Declarando variaveis
    #---------------------------------------------------------------------------
    
    i= declare("int", 1)
    
    DE = declare("matrix(36)") #matriz elástica vetorizada
    matrix_zero(DE, 36)

    stress_inc = declare("matrix(6)") #incremento de tensões
    stress_inc_flat = declare("matrix(9)") #incremento de tensões
    stressI = declare("matrix(6)") #tensões após previsão elástica
    stressI_flat = declare("matrix(9)") #tensões após previsão elástica
    eigvalsI = declare("matrix(3)") #tensões principais iniciais

    #---------------------------------------------------------------------------
    #Passando variaveis
    #---------------------------------------------------------------------------

    #Passando variáveis de estado para variáveis locais
    ocr = stVar0[1] #Passando OCR
    e   = stVar0[2] #Passando índice de vazios
    y   = stVar0[3] #Passando o parâmetro de estado
    p0  = stVar0[4] #Passando tensão de pré-adensamento
    v   = stVar0[6] #Passando índice de vazios específico
    

    #Passando propriedades para variaveis locais, Variável
    kappa   =  props[0]  # k
    nu      =  props[1]  # nu
    gamma   =  props[2]  # gamma
    _lambda =  props[3]  # lambda
    n       =  props[4]  # n, não confundir com pN que a tensão média nova
    r       =  props[5]  # r
    m       =  props[6]  # m
    phi     =  props[7]  # ângulo de atrito em graus
    p1      =  props[8]  # tensão de referência para gamma
    ratioPS =  props[9]  # controlador de psmall
    lode    =  props[10] # habilita a variação do ângulo de Lode, se for diferente de zero
    psmall  =  props[18] #tensão mínima a ser considerada
    Mtc     = props [19] #M na condição de compressão triaxial
    a       = props [20] #parâmetro usado na equação de Mteta

    #---------------------------------------------------------------------------
    #Plastificação inicial
    #---------------------------------------------------------------------------
    
    yield_0 = stVar0[14]
    Mteta0 = stVar0[11]

    yield_save = yield_0

    p_0 = -(stress0_flat[0] + stress0_flat[4] + stress0_flat[8])/3.0

    #---------------------------------------------------------------
    #Iteracoes
    #---------------------------------------------------------------
    
    c = 0.0 #contador para controle
    c1 = 0.0
    c2 = 0.0
    yieldN = 1.0

    while c < maxits[0]:
        c += 1.0
        c1 = 0.0
        dpegasus = (pegasus1[0] - pegasus0[0]) / nsub[0]

        while c1 < nsub[0]:
            c1 += 1.0
            c2 += 1.0
            pegasus[0] = pegasus0[0] + dpegasus 

            #Matriz elástica 
            evInc = - pegasus[0] * (eps[0] + eps[1] + eps[2])
            bulk = p_0 * (exp (v * evInc / kappa) - 1.0)/ evInc 
            if evInc == 0.0: bulk = v * p_0 / kappa 
            if bulk < 10e3 * p1: bulk = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
            if bulk > 100e6 * p1: bulk = 100e6 * p1  #limite superior bulk=100 GPa
            shear = 1.5 * bulk * (1.0 - 2.0 * nu) / (1.0 + nu) 
            alpha1 = bulk + 4.0 * shear / 3.0
            alpha2 = bulk - 2.0 * shear / 3.0

            DE [0]  = alpha1
            DE [7]  = alpha1
            DE [14] = alpha1
            DE [1]  = alpha2
            DE [2]  = alpha2
            DE [6]  = alpha2
            DE [8]  = alpha2
            DE [12] = alpha2
            DE [13] = alpha2
            DE [21] = 2.0 * shear
            DE [28] = 2.0 * shear
            DE [35] = 2.0 * shear
            
            #Previsao Elastica
            matrix_multiply_vector(DE, eps, stress_inc, 6) #multiplicação stress_inc = DE * eps
            voight_to_flat (stress_inc, stress_inc_flat)
            for i in range (9):
                stressI_flat[i] = stress0_flat[i] + stress_inc_flat[i] * pegasus[0]

            #Verficação de plastificação F
            eig_vals_3x3_analytical_erick(stressI_flat, eigvalsI)
            s1_I = - eigvalsI[0] #compressão positivo
            s2_I = - eigvalsI[1] 
            s3_I = - eigvalsI[2] 
            
            #Calculando os invariantes de cambrigde
            pI = (s1_I + s2_I + s3_I) / 3.0    
            qI = sqrt(((s1_I - s2_I)**2.0) + ((s2_I - s3_I)**2.0) + 
                ((s3_I - s1_I)**2.0)) / sqrt(2.0)
   
            #Verficação de plastificação 
            yieldN = ((qI**n) / ((Mteta0 * pI)**n) + 
                    log(pI / p0) / log(r))
            
            if yieldN > Ftol[0]:
                pegasus1[0] = pegasus[0]
                if yield_0 < -Ftol[0]:
                    #pegaus unload finalziado
                    c1 = nsub[0] + 1.0 #sai do loop 2
                    c = maxits[0] + 1.0 #sai do loop 2
                    #printf("\nentrei aq1")
                else:
                    pegasus0[0] = 0.0
                    yield_0 = yield_save
                    c1 = nsub[0] + 1.0 #sai do loop 2
                    #printf("\nentrei aq2")
            else:
                pegasus0[0] = pegasus[0]
                yield_0 = yieldN
                #printf("\nentrei aq3")

    if pegasus1[0] > 1.0 or pegasus0[0] < 0.0 or pegasus0[0]>pegasus1[0]:
        if pegasus1[0]>1.0:pegasus1[0] = 1.0
        if pegasus0[0]<0.0:pegasus1[0] = 0.0
        if pegasus0[0]>pegasus1[0]:
            pegasus1[0] = 1.0
            pegasus1[0] = 0.0
        '''
        printf("\nPegasusUnload_Print1")
        printf("\nid\n")
        printf("%d\n",idx)
        printf("c\n")
        printf("%.16e\n",c)
        printf("c1\n")
        printf("%.16e\n",c1)
        printf("c2\n")
        printf("%.16e\n",c2)
        printf("pegasus[0]\n")
        printf("%.16e\n",pegasus[0])
        printf("pegasus0[0]\n")
        printf("%.16e\n",pegasus0[0])
        printf("pegasus1[0]\n")
        printf("%.16e\n",pegasus1[0])
        printf("yieldN\n")
        printf("%.16e\n",yieldN)
        printf("yield_0\n")
        printf("%.16e\n",yield_0)
        printf("yield_save\n")
        printf("%.16e\n",yield_save)
        '''
