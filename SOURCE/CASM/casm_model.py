#Imports
from math import (pi, fabs, sqrt, log, pow, exp, tan, sin, cos, acos, isnan,
                  atan)

from matrix_operations import (eig_vals_3x3_analytical_erick,
                                matrix_multiply_vector, flat_to_voight,
                               voight_to_flat, matrix_transpose,
                               matrix_multiply, matrix_eigenvectors,
                               matrix_eigenvectors_erick, matrix_zero)

from compyle.api import declare

#ef = declare("int")
#i = [0]
#i = declare("matrix(1)")

def casm_model(props = [0], stress0_flat = [0], stVar0 = [0], eps = [0],
               yield1 = [0], stressN_flat = [0], stVarN = [0], yield2 = [0],
               Ftol = [0], lambdaS = [0], yield_flag = [0], eps_flat = [0],
               idx=0, mod = 1.0):
    
    #preciso otimizar o código para evitar cálculos repetidos, especialmente na
    # parte da previsão elástica, talvez usar um booleano que faz com que o
    # código passe pela porção elástica

    #i= 10
    #---------------------------------------------------------------------------
    #Declarando variaveis
    #---------------------------------------------------------------------------
    prt = 0.0
    i, j = declare("int", 2)
    DE = declare ("matrix(36)") #matriz elástica vetorizada
    
    eigvals0 = declare("matrix(3)") #tensões principais iniciais
    eigvecs0 = declare("matrix(9)") #direções principais iniciais
    stress_inc = declare("matrix(6)") #incremento de tensões
    stress_inc_flat = declare("matrix(9)") #incremento de tensões
    eigvals_inc = declare("matrix(3)") #tensões principais do incremento de tensões
    stressI = declare("matrix(6)") #tensões após previsão elástica
    stressI_flat = declare("matrix(9)") #tensões após previsão elástica
    eigvalsI = declare("matrix(3)") #tensões principais após previsão elástica
    eigvecsI = declare("matrix(9)") #direções principais após previsão elástica
    cosAng = 1.0 #ângulo para verificação de carregamento
    dp0 = 0.0 #incremento da tensão de plastificação, para endurecimento
    matA = declare("matrix(9)") #matrix A
    matAT = declare("matrix(9)") #matrix A transposta
    matAux = declare("matrix(9)") #matrix auxiliar para cálculos intermediários
    epsMain_flat = declare("matrix(9)") #matrix com deformações nas direções principais

    #zerando matriz
    matrix_zero(DE, 36)
    matrix_zero(eigvals0, 3)
    matrix_zero(eigvecs0, 9)
    matrix_zero(stress_inc, 6)
    matrix_zero(stress_inc_flat, 9)
    matrix_zero(eigvals_inc, 3)
    matrix_zero(stressI, 6)
    matrix_zero(stressI_flat, 9)
    matrix_zero(eigvalsI, 3)
    matrix_zero(eigvecsI, 9)
    matrix_zero(matAT, 9)   
    matrix_zero(matA, 9)
    matrix_zero(matAux, 9)
    matrix_zero(epsMain_flat, 9)

    #---------------------------------------------------------------------------
    #Passando variaveis
    #---------------------------------------------------------------------------

    #Para controle de erros em condicao isotropica
    '''
    ver se isso vai ser preciso aq, no Plaxis era preciso pq a função não conseguia retornar s1,s2 e s3 quando era uma condição isotrópica
    If (abs(Sig0(2) - Sig0(3)) .LT. 0.000000001D0.AND.
    *    abs(Sig0(2) - Sig0(1)) .LT. 0.000000001D0)
    *     Sig0(2) = Sig0(2) - 0.0001D0
    ! Para controle de erros em condição isotrópica
    '''

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
    #Cálculo do estado de tensões inicial
    #---------------------------------------------------------------------------
    
    #O cálculo desse bloco poderia ser levado para dentro dp else quando plástico, acho que não uso ele pra nada aqui fora

    #Calculando tensões principais iniciais (0)
    if (stress0_flat[0]-stress0_flat[8])==0.0: stress0_flat[0] -=0.00001
    eig_vals_3x3_analytical_erick(stress0_flat, eigvals0)
    s1_0 = - eigvals0[0] #compressão positivo
    s2_0 = - eigvals0[1] 
    s3_0 = - eigvals0[2] 

    #Calculando os invariantes de cambrigde iniciais (0)
    p_0 = (s1_0 + s2_0 + s3_0) / 3.0    
    q_0 = sqrt(((s1_0 - s2_0)**2.0) + ((s2_0 - s3_0)**2.0) + 
            ((s3_0 - s1_0)**2.0)) / sqrt(2.0)
    #teta0 = ((atan((s1_0 - 2.0 * s2_0 + s3_0) / 
    #            (sqrt(3.0)*(s1_0 - s3_0)))) * 180.0 / pi)
    #teta0 = atan((s1_0 - 2.0 * s2_0 + s3_0) / 
    #            (sqrt(3.0)*(s1_0 - s3_0)))
    Mteta0 = 0.0
    teta0 = 0.0
    if lode == 0.0: 
        teta0 = 30.0 #Mlode = Mtc
        Mteta0 = Mtc
    else:
        if fabs(s1_0 - s3_0) < psmall: 
            teta0 = 30.0 #condição isotrópica
            Mteta0 = Mtc  
        else:
            teta0 = ((atan((s1_0 - 2.0 * s2_0 + s3_0) / 
                   (sqrt(3.0)*(s1_0 - s3_0)))) * 180.0 / pi)  
            Mteta0 = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                    (sin(-3.0 * teta0 * pi / 180.0))))**(1.0 / 4.0)
    
    
    #---------------------------------------------------------------
    #Previsão Elástica
    #---------------------------------------------------------------
    
    #o cálculo desse bloco deveria ser acionado apenas quando se precisa fazer aprevisão elástica, posso pensar nisso por meio de um flag
    
    #Matriz elástica
    evInc1 = - (eps[0] + eps[1] + eps[2]) #incremento de deformacaovolumetrica, compressão positivo
    #bulk = p_0 * (exp (v * evInc1 / kappa) - 1.0)/ evInc1 #obs: não me recordo de como cheguei nessa equação, essa formulação considera já uma previsão devido a deformação, não sei o quanto que isso melhora em relação a equação a baixo.
    bulk = v * p_0 / kappa
    #if evInc1 == 0.0: bulk = v * p_0 / kappa #
    if bulk < 10e3 * p1: bulk = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
    if bulk > 100e6 * p1: bulk = 100e6 * p1  #limite superior bulk=100 GPa
    #young = 20000000
    #bulk = young / (3.0 * (1.0 - 2.0*nu))
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
    DE [21] = 2*shear
    DE [28] = 2*shear
    DE [35] = 2*shear
    
    #Incremento de tensões
    matrix_multiply_vector(DE, eps, stress_inc, 6) #multiplicação stress_inc = DE * eps
    voight_to_flat (stress_inc, stress_inc_flat)
    for i in range (9):
        stressI_flat[i] = stress0_flat[i] + stress_inc_flat[i] 

    #Calculando tensões principais após a previsão elástica (I)
    eig_vals_3x3_analytical_erick(stressI_flat, eigvalsI)
    s1_I = - eigvalsI[0] #compressão positivo
    s2_I = - eigvalsI[1] 
    s3_I = - eigvalsI[2] 

    #Calculando os invariantes de cambrigde iniciais (0)
    pI = (s1_I + s2_I + s3_I) / 3.0    
    qI = sqrt(((s1_I - s2_I)**2.0) + ((s2_I - s3_I)**2.0) + 
            ((s3_I - s1_I)**2.0)) / sqrt(2.0)
    tetaI = 0.0
    MtetaI = 0.0
    

    #Controle de material superficial
    if pI < psmall:
        pI = psmall
        qI = 0.0 #inserir isso agora
        tetaI = 30.0

    #Cálculo do Mteta
    if lode == 0.0: 
        tetaI = 30.0 #Mlode = Mtc
        MtetaI = Mtc
    else:
        if fabs(s1_I - s3_I) < psmall: #condição isotrópica
            tetaI = 30.0 
            MtetaI = Mtc  
        else:
            tetaI = ((atan((s1_I - 2.0 * s2_I + s3_I) / 
                   (sqrt(3.0)*(s1_I - s3_I)))) * 180.0 / pi)  
            MtetaI = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                    (sin(-3.0 * tetaI * pi / 180.0))))**(1.0 / 4.0)

    #for i in range (9):
    #    stressN_flat[i] = stressI_flat[i]

    #Verficação de plastificação
    yield1[0] = ((qI**n) / ((MtetaI * pI)**n) + 
                log(pI / p0) / log(r))   
    if yield2[0] == -1.0: #garante que alguns parâmetros sejam alterados só na primeira passada
        yield2[0] = yield1[0]#guarda o erro inicial
        #if yield1[0] <= - Ftol[0]:
            #stVarN[0] = -1.0 #indicador externo que material está elástico (atual)
            #d_flag = 0 #Atualizar d_flag aqui, mas preciso entender melhor essa variável
        if isnan(yield2[0]):
            if prt == 1.0:
                printf("\nERRO_yield2[0]:\n")
                printf("%.16e\n",yield2[0])
                printf("\nParticula:")
                printf("%d\n",idx)
            yield2[0] = 0.0
    #---------------------------------------------------------------------------
    #Teste de plastificacao
    #---------------------------------------------------------------------------

    s1_N = 0.0
    s2_N = 0.0
    s3_N = 0.0
    pN = 0.0 #tensão média nova, não confundir com n (parâmetro n)
    qN = 0.0
    tetaN = 0.0
    MtetaN = 0.0
    er = 1
    #if mod == 1 and yield1[0] <= Ftol[0]: #criei isso nessa versão
    if yield1[0] <= Ftol[0]:
    #if yield1[0] <= Ftol[0] or pI<=psmall or pN <= psmall:
    #if er == 0:
        #Elástico
        #preciso separar se elástico ou plástico mas ainda sobre a superfície (sem necessidade de correção)
        if stVar0[0] == 1: 
            stVar0[0] = 0 #indicador externo que material está elástico (atual), mas plastico (passado)
        else:
            stVar0[0] = -1.0 #contia elásticocd sph/02_PySPH_Code_Erick
        
        yield_flag[0] = 0.0 #indicador interno que material está elástico, pula a correção plástica, equivale ao i_yield no plaxis
        s1_N   = s1_I
        s2_N   = s2_I
        s3_N   = s3_I
        pN    = pI #tensão média nova, não confundir com n (parâmetro n)
        qN    = qI
        tetaN = tetaI
        MtetaN = MtetaI
        for i in range (9):
            stressN_flat[i] = stressI_flat[i]

    #elif mod == 0:
    else:
        #Plástico
        stVarN[0] = 1.0 #indicador externo que material está plástico (atual)
        yield_flag[0] = 1.0 #indicador interno que material está plástico, entra na correção plástica

        #Derivadas da superfície de plastificação e da lei de fluxo são feitas utilizando as tensões iniciais (0)
        #obs: acho que o bloco 1 pode vim pra cá
        
        s1_N = s1_0
        s2_N = s2_0
        s3_N = s3_0
        pN = p_0 #tensão média nova, não confundir com p_n (parâmetro n)
        qN = q_0
        tetaN = teta0
        MtetaN = Mteta0

       
        #-------------------------------------------
        #Derivadas da superfície de plastificação
        #-------------------------------------------
       
        fa = 0.0 #df/ds1
        fb = 0.0 #df/ds2
        fc = 0.0 #df/ds3

        fax1 = (MtetaN**n) * ((3.0 * pN)**(n - 1.0))
        fax2 = (3.0**n) * log(r)
        fax3 = (n * log(pN / p0)) + 1.0
        fax4 = n / 2.0
        fax5 = ((s1_N**2.0) + (s2_N**2.0) + (s3_N**2.0) -
                s1_N*s2_N - s1_N*s3_N - s2_N*s3_N)
        if fax5 < 0.0: fax5 = 0.0
        fax6 = fax5**((n - 2.0) / 2.0)

        fa = (fax1 * fax3 / fax2) + (fax4 * fax6) * (2.0 * s1_N - s2_N - s3_N) 
        fb = (fax1 * fax3 / fax2) + (fax4 * fax6) * (2.0 * s2_N - s1_N - s3_N) 
        fc = (fax1 * fax3 / fax2) + (fax4 * fax6) * (2.0 * s3_N - s1_N - s2_N) 
        
        #-------------------------------------------
        #Derivadas da funcao potencial plastica
        #-------------------------------------------
        
        ga = 0.0 #dg/ds1
        gb = 0.0 #dg/ds2
        gc = 0.0 #dg/ds3

        if m == -1.0: #lei de fluxo associada
            ga = fa
            gb = fb
            gc = fc
        else:
            pc = ((pN / (m - 1.0)) * (((qN / (MtetaN * pN))**m) - 
                1.0 + m))

            gax1 = (m * (MtetaN**m) * ((3.0 * pN)**(m - 2.0)))
            gax2 = 3.0**m
            gax3 = 3.0 * pN * (m - 1.0)
            gax4 = 3.0 * pc * (2.0 - m - (1.0 / m))
            gax5 = m / 2.0
            gax6 = ((s1_N**2.0) + (s2_N**2.0) + (s3_N**2.0) -
                    s1_N*s2_N - s1_N*s3_N - s2_N*s3_N)
            if gax6 < 0.0: gax6 = 0.0
            gax7 = gax6**((m - 2.0) / 2.0)

            ga = (gax1 * (gax3 + gax4) / gax2 + (gax5 * gax7) * 
                (2.0 * s1_N - s2_N - s3_N))
            gb = (gax1 * (gax3 + gax4) / gax2 + (gax5 * gax7) * 
                (2.0 * s2_N - s1_N - s3_N))
            gc = (gax1 * (gax3 + gax4) / gax2 + (gax5 * gax7) * 
                (2.0 * s3_N - s1_N - s2_N))
    
        #-------------------------------------------
        #Matriz de correcao plastica
        #-------------------------------------------
        
        alpha = (- (-(1.0 + e) * (ga + gb + gc) / 
                ((_lambda - kappa) * log(r))))
        
        beta = alpha + (
                        (fa * alpha1 + fb * alpha2 + fc * alpha2)*ga + 
                        (fa * alpha2 + fb * alpha1 + fc * alpha2)*gb +
                        (fa * alpha2 + fb * alpha2 + fc * alpha1)*gc ) 
        
        bGa = alpha1 * ga + alpha2 * gb + alpha2 * gc
        bGb = alpha2 * ga + alpha1 * gb + alpha2 * gc
        bGc = alpha2 * ga + alpha2 * gb + alpha1 * gc

        bFa = alpha1 * fa + alpha2 * fb + alpha2 * fc
        bFb = alpha2 * fa + alpha1 * fb + alpha2 * fc
        bFc = alpha2 * fa + alpha2 * fb + alpha1 * fc
    
        #-------------------------------------------
        #Correcao plastica
        #-------------------------------------------

        
        #Calculando as direcoes principais
        matrix_eigenvectors_erick(stressI_flat,eigvalsI,matA,3)
        #usando as direcoes principais da previsao elástica, fique na dúvida se não deveria usar stress0_flat

        #Calculando os incrementos de deformações nas direções principais
        #matA = eigvecsI #[A]
        matrix_transpose(matA, matAT, 3) #[A]T
        matrix_multiply (matA, eps_flat, matAux, 3) #[A][eps]
        matrix_multiply (matAux, matAT, epsMain_flat, 3) #[A][eps][A]T

    
        e1 = - epsMain_flat [0] #deformação na direação de s1, compressão positivo
        e2 = - epsMain_flat [4] #deformação na direação de s2, compressão positivo
        e3 = - epsMain_flat [8] #deformação na direação de s3, compressão positivo

        #lambdaS[0] = ((bFa * e1 + bFb * e2 + bFc * e3) / beta)
        if beta == 0.0:
            lambdaS[0] = 0.0
        else: 
            lambdaS[0] = ((bFa * e1 + bFb * e2 + bFc * e3) / beta)  
            if lambdaS[0] < 0.0: lambdaS[0] = 0.0
        
        if isnan(lambdaS[0]):
            if prt == 1.0:
                printf("\nERROlambdaS[0]:\n")
                printf("%.16e\n",lambdaS[0])
                printf("\nParticula:\n")
                printf("%d\n",idx)
            lambdaS[0] = 0.0

        if isnan(bGa or bGb or bGc):
            if prt == 1.0:
                printf("\nERRObGa:\n")
                printf("%.16e\n",bGa)
                printf("\nERRObGb:\n")
                printf("%.16e\n",bGb)
                printf("\nERRObGc:\n")
                printf("%.16e\n",bGc)
                printf("\nParticula:\n")
                printf("%d\n",idx)
            bGa = 0.0
            bGb = 0.0
            bGc = 0.0

        s1_N = s1_I - bGa * lambdaS[0]
        s2_N = s2_I - bGb * lambdaS[0]
        s3_N = s3_I - bGc * lambdaS[0]

        pN = (s1_N + s2_N + s3_N) / 3.0    
        qN = sqrt(((s1_N - s2_N)**2.0) + ((s2_N - s3_N)**2.0) + 
            ((s3_N - s1_N)**2.0)) / sqrt(2.0)
  
        #Controle de material superficial
        if pN < psmall: 
            pN = psmall
            tetaN = 30.0
            qN = 0.0 #inseri isso agora   
    
        #Cálculo do Mteta
        if lode == 0.0: 
            tetaN = 30.0 #Mlode = Mtc
            MtetaN = Mtc
        else:
            if fabs(s1_N - s3_N) < psmall: #condição isotrópica
                tetaN = 30.0 
                MtetaN = Mtc  
            else:
                tetaN = atan((s1_N - 2.0 * s2_N + s3_N) / 
                        (sqrt(3.0)*(s1_N - s3_N))) * 180.0 / pi  
                MtetaN = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                        (sin(-3.0 * tetaN * pi / 180.0))))**(1.0 / 4.0)

        #Aplicando correção, baseado na forma como o FLAC faz no modelo Cam-Clay, divide-se a parte desviadora da hisrostática


        if qI == 0.0: #condição isotrópica
            for i in range (9):
                stressN_flat[i] = stressI_flat[i]
            correction = pI - pN #correction compressão é positivo
            stressN_flat[0] += correction #stressN_flat compressão é negativo ( por isso soma corretion)
            stressN_flat[4] += correction #compressão positivo
            stressN_flat[8] += correction #compressão positivo   
        else:
            dVal = qN / qI
            
            for i in range (9):
                stressN_flat [i] = stressI_flat [i] * dVal 
            
            stressN_flat[0] = (stressI_flat [0] + pI) * dVal - pN #compressão positivo, subrai a hidrotática, corrige o desvio, adciona o hidrostático corrgio; stressN_flat compressão é negativo ( por isso inverte sinal)
            stressN_flat[4] = (stressI_flat [4] + pI) * dVal - pN 
            stressN_flat[8] = (stressI_flat [8] + pI) * dVal - pN 

        #-------------------------------------------
        #Endurecimento
        #-------------------------------------------
        
        b = - alpha * (- p0 * log(r))
        dp0 = b * lambdaS[0]
        if isnan(dp0):
            if prt == 1.0:
                printf("\nERROdp0:\n")
                printf("%.16e\n",dp0)
                printf("\nERROb:\n")
                printf("%.16e\n",b)
                printf("\nParticula:\n")
                printf("%d\n",idx)
            dp0 = 0.0
        p0 = p0 + dp0
        #p0 = (pN * exp(log(r) * ((qN / (MtetaN * pN))**n)))
        
        #outra forma possível

        #-------------------------------------------
        #Verificação de carregamento
        #-------------------------------------------
        
        #voight_to_flat (stress_inc, stress_inc_flat) #incremento de tensões na previsão elástica
        eig_vals_3x3_analytical_erick(stress_inc_flat, eigvals_inc)
        dsig1 = - eigvals_inc[0]
        dsig2 = - eigvals_inc[1]
        dsig3 = - eigvals_inc[2]
        aux1 = dsig1 * dsig1 + dsig2 * dsig2 + dsig3 * dsig3 #módulo de eigvals_inc
        aux2 = fa * fa + fb * fb + fc * fc #módulo de {aF}
        aux3 = sqrt (aux1 * aux2)
        aux4 = fa * dsig1 + fb * dsig2 + fc * dsig3 #produto interno
        if aux3 == 0.0: #evita erro quando não há mudança de tensão
            cosAng = 2.0 #só pra saber que entrou aqui e passar a informação como carregamento 
        else:   
            cosAng = aux4 / aux3 #cosAng > 0 carrega
        if isnan(cosAng):
            if prt == 1.0: 
                printf("\nERROcosAng:\n")
                printf("%.16e\n",cosAng)
                printf("\nParticula:\n")
                printf("%d\n",idx)
            cosAng = 2.0
    #-------------------------------------------
    #Controle para baixas tensões
    #-------------------------------------------
    
    if p0 <= psmall or pN <= psmall:
        #Plastificou
        stVarN[0] = 1.0 #indicador externo que material está plástico (atual)
        yield_flag[0] = 0.0 #apenas para pular o loop de correção plástica
        #for i in range (9):
        #    stressN_flat [i] = stressI_flat [i]
            #stressN_flat [i] = 0.0
            #if i % 4 == 0: stressN_flat[i] = - psmall # Hydrostatic components
        
        s1_N   = psmall
        s2_N   = psmall
        s3_N   = psmall
        pN     = psmall
        qN     = 0.0
        tetaN  = 30.0
        MtetaN = Mtc
        cosAng = 1.0
        if ocr < 1.0: ocr = 1.0
        if p0 < psmall: 
            p0 = psmall
            ocr = 1.0
        if ocr == 1.0: p0 = psmall

        #inseri isso agora
        #k0 = nu / (1.0-nu)
        #qN = pN * 3.0 * (1.0-k0)/(1.0+2.0*k0)
        #qN = pN * MtetaN
        v = gamma
        ev = 1.0 - v / stVar0 [13]
    else:
        p0aux = (pN * exp(log(r) * ((qN / (MtetaN * pN))**n)))
        ocr = p0 / p0aux
        yield1[0] = ((qN**n) / ((MtetaN * pN)**n) + log(pN / p0) / 
                    log(r))
        evInc1 = - (eps[0] + eps[1] + eps[2])
        ev = stVar0 [5] + evInc1 #deformação volumétrica acumulada
        v = stVar0 [13] * (1.0 - ev) #referecnial inicial
    #s_v = s_v * (1.0 - evInc1) #referencial alterado constantemente
    
    #------------------------------------------------
    #Alteração de variáveis de estado
    #------------------------------------------------
  
    e = v - 1.0
    y = v + _lambda * log(pN / p1) - gamma
    

    if isnan(yield1[0]):
        if prt == 1.0:
            printf("\nERRO_yield1 [0]:\n")
            printf("%.16e\n",yield1[0])
            printf("\nParticula:")
            printf("%d\n",idx)
        yield1[0] = 0.0


    #stVarN[0] = dp0
    stVarN[1] = ocr
    #printf("%.1f\n", ocr)
    stVarN[2] = e
    #printf("%.1f\n", e)
    stVarN[3] = y
    stVarN[4] = p0
    stVarN[5] = ev
    #printf("%.12f\n", stVar0 [5])
    #printf("%.12f\n", ev)
    #printf("%.12f\n", evInc1)
    stVarN[6] = v
    #printf("%.1f\n", v)
    stVarN[7] = qN
    stVarN[8] = pN
    stVarN[9] = qN / pN
    stVarN[10] = tetaN
    stVarN[11] = MtetaN
    stVarN[12] = qN / (pN * MtetaN)
    stVarN[13] = stVar0[13]
    stVarN[14] = yield1[0]
    stVarN[15] = bulk
    stVarN[16] = shear
    stVarN[17] = cosAng

    for i in range (18):
        if isnan(stVarN[i]):
            if prt == 1.0:
                printf("\nERRO_stVarN:\n")
                printf("%d\n",i)
                printf("%.16e\n",stVarN[i])
                printf("\nParticula:")
                printf("%d\n",idx)
            stVarN[i] = stVar0[i]
    
    for i in range (9):
        if isnan(stressN_flat[i]):
            if prt == 1.0:
                printf("\nERRO_stressN_flat:\n")
                printf("%d\n",i)
                printf("%.16e\n",stressN_flat[i])
                printf("\nParticula:")
                printf("%d\n",idx)
            stressN_flat[i] = stress0_flat[i]


def casm_model_EL(props = [0], stress0_flat = [0], stVar0 = [0], 
                        eps = [0], yield1 = [0], stressN_flat = [0], stVarN = [0], yield2 = [0], Ftol = [0], lambdaS = [0], yield_flag = [0], eps_flat = [0], idx=0, mod = 1.0):
    
    #preciso otimizar o código para evitar cálculos repetidos, especialmente na parte da previsão elástica, talvez usar um bolleano que faz com que o código passe pela porção elástica

    #i= 10
    #---------------------------------------------------------------------------
    #Declarando variaveis
    #---------------------------------------------------------------------------
    prt = 0.0
    i, j = declare("int", 2)
    DE = declare ("matrix(36)") #matriz elástica vetorizada
    
    eigvals0 = declare("matrix(3)") #tensões principais iniciais
    eigvecs0 = declare("matrix(9)") #direções principais iniciais
    stress_inc = declare("matrix(6)") #incremento de tensões
    stress_inc_flat = declare("matrix(9)") #incremento de tensões
    eigvals_inc = declare("matrix(3)") #tensões principais do incremento de tensões
    stressI = declare("matrix(6)") #tensões após previsão elástica
    stressI_flat = declare("matrix(9)") #tensões após previsão elástica
    eigvalsI = declare("matrix(3)") #tensões principais após previsão elástica
    eigvecsI = declare("matrix(9)") #direções principais após previsão elástica
    cosAng = 1.0 #ângulo para verificação de carregamento
    dp0 = 0.0 #incremento da tensão de plastificação, para endurecimento
    matA = declare("matrix(9)") #matrix A
    matAT = declare("matrix(9)") #matrix A transposta
    matAux = declare("matrix(9)") #matrix auxiliar para cálculos intermediários
    epsMain_flat = declare("matrix(9)") #matrix com deformações nas direções principais

    #zerando matriz
    matrix_zero(DE, 36)
    matrix_zero(eigvals0, 3)
    matrix_zero(eigvecs0, 9)
    matrix_zero(stress_inc, 6)
    matrix_zero(stress_inc_flat, 9)
    matrix_zero(eigvals_inc, 3)
    matrix_zero(stressI, 6)
    matrix_zero(stressI_flat, 9)
    matrix_zero(eigvalsI, 3)
    matrix_zero(eigvecsI, 9)
    matrix_zero(matAT, 9)   
    matrix_zero(matA, 9)
    matrix_zero(matAux, 9)
    matrix_zero(epsMain_flat, 9)

    #---------------------------------------------------------------------------
    #Passando variaveis
    #---------------------------------------------------------------------------

    #Para controle de erros em condicao isotropica
    '''
    ver se isso vai ser preciso aq, no Plaxis era preciso pq a função não conseguia retornar s1,s2 e s3 quando era uma condição isotrópica
    If (abs(Sig0(2) - Sig0(3)) .LT. 0.000000001D0.AND.
    *    abs(Sig0(2) - Sig0(1)) .LT. 0.000000001D0)
    *     Sig0(2) = Sig0(2) - 0.0001D0
    ! Para controle de erros em condição isotrópica
    '''

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
    #Cálculo do estado de tensões inicial
    #---------------------------------------------------------------------------
    
    #O cálculo desse bloco poderia ser levado para dentro dp else quando plástico, acho que não uso ele pra nada aqui fora

    #Calculando tensões principais iniciais (0)
    if (stress0_flat[0]-stress0_flat[8])==0.0: stress0_flat[0] -=0.00001
    eig_vals_3x3_analytical_erick(stress0_flat, eigvals0)
    s1_0 = - eigvals0[0] #compressão positivo
    s2_0 = - eigvals0[1] 
    s3_0 = - eigvals0[2] 

    #Calculando os invariantes de cambrigde iniciais (0)
    p_0 = (s1_0 + s2_0 + s3_0) / 3.0    
    q_0 = sqrt(((s1_0 - s2_0)**2.0) + ((s2_0 - s3_0)**2.0) + 
            ((s3_0 - s1_0)**2.0)) / sqrt(2.0)
    #teta0 = ((atan((s1_0 - 2.0 * s2_0 + s3_0) / 
    #            (sqrt(3.0)*(s1_0 - s3_0)))) * 180.0 / pi)
    #teta0 = atan((s1_0 - 2.0 * s2_0 + s3_0) / 
    #            (sqrt(3.0)*(s1_0 - s3_0)))
    Mteta0 = 0.0
    teta0 = 0.0
    if lode == 0.0: 
        teta0 = 30.0 #Mlode = Mtc
        Mteta0 = Mtc
    else:
        if fabs(s1_0 - s3_0) < psmall: 
            teta0 = 30.0 #condição isotrópica
            Mteta0 = Mtc  
        else:
            teta0 = ((atan((s1_0 - 2.0 * s2_0 + s3_0) / 
                   (sqrt(3.0)*(s1_0 - s3_0)))) * 180.0 / pi)  
            Mteta0 = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                    (sin(-3.0 * teta0 * pi / 180.0))))**(1.0 / 4.0)
    
    
    #---------------------------------------------------------------
    #Previsão Elástica
    #---------------------------------------------------------------
    
    #o cálculo desse bloco deveria ser acionado apenas quando se precisa fazer aprevisão elástica, posso pensar nisso por meio de um flag
    
    #Matriz elástica
    evInc1 = - (eps[0] + eps[1] + eps[2]) #incremento de deformacaovolumetrica, compressão positivo
    #bulk = p_0 * (exp (v * evInc1 / kappa) - 1.0)/ evInc1 #obs: não me recordo de como cheguei nessa equação, essa formulação considera já uma previsão devido a deformação, não sei o quanto que isso melhora em relação a equação a baixo.
    bulk = v * p_0 / kappa
    #if evInc1 == 0.0: bulk = v * p_0 / kappa #
    if bulk < 10e3 * p1: bulk = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
    if bulk > 100e6 * p1: bulk = 100e6 * p1  #limite superior bulk=100 GPa
    #young = 20000000
    #bulk = young / (3.0 * (1.0 - 2.0*nu))
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
    DE [21] = 2*shear
    DE [28] = 2*shear
    DE [35] = 2*shear
    
    #Incremento de tensões
    matrix_multiply_vector(DE, eps, stress_inc, 6) #multiplicação stress_inc = DE * eps
    voight_to_flat (stress_inc, stress_inc_flat)
    for i in range (9):
        stressI_flat[i] = stress0_flat[i] + stress_inc_flat[i] 

    #Calculando tensões principais após a previsão elástica (I)
    eig_vals_3x3_analytical_erick(stressI_flat, eigvalsI)
    s1_I = - eigvalsI[0] #compressão positivo
    s2_I = - eigvalsI[1] 
    s3_I = - eigvalsI[2] 

    #Calculando os invariantes de cambrigde iniciais (0)
    pI = (s1_I + s2_I + s3_I) / 3.0    
    qI = sqrt(((s1_I - s2_I)**2.0) + ((s2_I - s3_I)**2.0) + 
            ((s3_I - s1_I)**2.0)) / sqrt(2.0)
    tetaI = 0.0
    MtetaI = 0.0
    

    #Controle de material superficial
    if pI < psmall:
        pI = psmall
        qI = 0.0 #inserir isso agora
        tetaI = 30.0

    #Cálculo do Mteta
    if lode == 0.0: 
        tetaI = 30.0 #Mlode = Mtc
        MtetaI = Mtc
    else:
        if fabs(s1_I - s3_I) < psmall: #condição isotrópica
            tetaI = 30.0 
            MtetaI = Mtc  
        else:
            tetaI = ((atan((s1_I - 2.0 * s2_I + s3_I) / 
                   (sqrt(3.0)*(s1_I - s3_I)))) * 180.0 / pi)  
            MtetaI = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                    (sin(-3.0 * tetaI * pi / 180.0))))**(1.0 / 4.0)

    #for i in range (9):
    #    stressN_flat[i] = stressI_flat[i]

    #Verficação de plastificação
    yield1[0] = ((qI**n) / ((MtetaI * pI)**n) + 
                log(pI / p0) / log(r))   
    if yield2[0] == -1.0: #garante que alguns parâmetros sejam alterados só na primeira passada
        yield2[0] = yield1[0]#guarda o erro inicial
        #if yield1[0] <= - Ftol[0]:
            #stVarN[0] = -1.0 #indicador externo que material está elástico (atual)
            #d_flag = 0 #Atualizar d_flag aqui, mas preciso entender melhor essa variável
        if isnan(yield2[0]):
            if prt == 1.0:
                printf("\nERRO_yield2[0]:\n")
                printf("%.16e\n",yield2[0])
                printf("\nParticula:")
                printf("%d\n",idx)
            yield2[0] = 0.0
    #---------------------------------------------------------------------------
    #Teste de plastificacao
    #---------------------------------------------------------------------------

    s1_N = 0.0
    s2_N = 0.0
    s3_N = 0.0
    pN = 0.0 #tensão média nova, não confundir com n (parâmetro n)
    qN = 0.0
    tetaN = 0.0
    MtetaN = 0.0
    er = 1
    #if mod == 1 and yield1[0] <= Ftol[0]: #criei isso nessa versão
    if yield1[0] <= Ftol[0]:
    #if yield1[0] <= Ftol[0] or pI<=psmall or pN <= psmall:
    #if er == 0:
        #Elástico
        #preciso separar se elástico ou plástico mas ainda sobre a superfície (sem necessidade de correção)
        if stVar0[0] == 1: 
            stVar0[0] = 0 #indicador externo que material está elástico (atual), mas plastico (passado)
        else:
            stVar0[0] = -1.0 #contia elásticocd sph/02_PySPH_Code_Erick
        
        yield_flag[0] = 0.0 #indicador interno que material está elástico, pula a correção plástica, equivale ao i_yield no plaxis
        s1_N   = s1_I
        s2_N   = s2_I
        s3_N   = s3_I
        pN    = pI #tensão média nova, não confundir com n (parâmetro n)
        qN    = qI
        tetaN = tetaI
        MtetaN = MtetaI
        for i in range (9):
            stressN_flat[i] = stressI_flat[i]
    else:
        yield_flag[0] = 1.0 #indicador interno que material está elástico, pula a correção plástica, equivale ao i_yield no plaxis
        s1_N   = s1_0
        s2_N   = s2_0
        s3_N   = s3_0
        pN    = p_0 #tensão média nova, não confundir com n (parâmetro n)
        qN    = q_0
        tetaN = teta0
        MtetaN = Mteta0
        for i in range (9):
            stressN_flat[i] = stress0_flat[i]


    #-------------------------------------------
    #Controle para baixas tensões
    #-------------------------------------------
    
    if p0 <= psmall or pN <= psmall:
        #Plastificou
        stVarN[0] = 1.0 #indicador externo que material está plástico (atual)
        yield_flag[0] = 0.0 #apenas para pular o loop de correção plástica
        #for i in range (9):
        #    stressN_flat [i] = stressI_flat [i]
            #stressN_flat [i] = 0.0
            #if i % 4 == 0: stressN_flat[i] = - psmall # Hydrostatic components
        
        s1_N   = psmall
        s2_N   = psmall
        s3_N   = psmall
        pN     = psmall
        qN     = 0.0
        tetaN  = 30.0
        MtetaN = Mtc
        cosAng = 1.0
        if ocr < 1.0: ocr = 1.0
        if p0 < psmall: 
            p0 = psmall
            ocr = 1.0
        if ocr == 1.0: p0 = psmall

        #inseri isso agora
        #k0 = nu / (1.0-nu)
        #qN = pN * 3.0 * (1.0-k0)/(1.0+2.0*k0)
        #qN = pN * MtetaN
        v = gamma
        ev = 1.0 - v / stVar0 [13]
    else:
        p0aux = (pN * exp(log(r) * ((qN / (MtetaN * pN))**n)))
        ocr = p0 / p0aux
        yield1[0] = ((qN**n) / ((MtetaN * pN)**n) + log(pN / p0) / 
                    log(r))
        evInc1 = - (eps[0] + eps[1] + eps[2])
        ev = stVar0 [5] + evInc1 #deformação volumétrica acumulada
        v = stVar0 [13] * (1.0 - ev) #referecnial inicial
    #s_v = s_v * (1.0 - evInc1) #referencial alterado constantemente
    
    #------------------------------------------------
    #Alteração de variáveis de estado
    #------------------------------------------------
  
    e = v - 1.0
    y = v + _lambda * log(pN / p1) - gamma
    

    if isnan(yield1[0]):
        if prt == 1.0:
            printf("\nERRO_yield1 [0]:\n")
            printf("%.16e\n",yield1[0])
            printf("\nParticula:")
            printf("%d\n",idx)
        yield1[0] = 0.0


    #stVarN[0] = dp0
    stVarN[1] = ocr
    #printf("%.1f\n", ocr)
    stVarN[2] = e
    #printf("%.1f\n", e)
    stVarN[3] = y
    stVarN[4] = p0
    stVarN[5] = ev
    #printf("%.12f\n", stVar0 [5])
    #printf("%.12f\n", ev)
    #printf("%.12f\n", evInc1)
    stVarN[6] = v
    #printf("%.1f\n", v)
    stVarN[7] = qN
    stVarN[8] = pN
    stVarN[9] = qN / pN
    stVarN[10] = tetaN
    stVarN[11] = MtetaN
    stVarN[12] = qN / (pN * MtetaN)
    stVarN[13] = stVar0[13]
    stVarN[14] = yield1[0]
    stVarN[15] = bulk
    stVarN[16] = shear
    stVarN[17] = cosAng

    for i in range (18):
        if isnan(stVarN[i]):
            if prt == 1.0:
                printf("\nERRO_stVarN:\n")
                printf("%d\n",i)
                printf("%.16e\n",stVarN[i])
                printf("\nParticula:")
                printf("%d\n",idx)
            stVarN[i] = stVar0[i]
    
    for i in range (9):
        if isnan(stressN_flat[i]):
            if prt == 1.0:
                printf("\nERRO_stressN_flat:\n")
                printf("%d\n",i)
                printf("%.16e\n",stressN_flat[i])
                printf("\nParticula:")
                printf("%d\n",idx)
            stressN_flat[i] = stress0_flat[i]