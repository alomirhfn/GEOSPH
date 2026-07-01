#Imports
from math import (pi, fabs, sqrt, log, pow, exp, tan, sin, cos, acos, isnan,
                  atan)

from matrix_operations import (eig_vals_3x3_analytical_erick,
                                matrix_multiply_vector, flat_to_voight,
                               voight_to_flat, matrix_transpose,
                               matrix_multiply, matrix_eigenvectors,
                               matrix_eigenvectors_erick, matrix_zero)

from compyle.api import declare

def correction(props = [0], stress0_flat = [0], stVar0 = [0],
               yield1 = [0], stressN_flat = [0], stVarN = [0], Ftol = [0],
               maxits = [0], idx=0):
    prt = 0.0
    #---------------------------------------------------------------------------
    #Declarando variaveis
    #---------------------------------------------------------------------------
    
    i = declare("int", 1)

    eigvals = declare("matrix(3)") #tensões principais 

    #---------------------------------------------------------------------------
    #Passando variaveis
    #---------------------------------------------------------------------------

    #Passando variáveis de estado para variáveis locais
    ocr = stVar0[1] #Passando OCR
    e   = stVar0[2] #Passando índice de vazios
    p0  = stVar0[4] #Passando tensão de pré-adensamento
    v   = stVar0[6] #Passando índice de vazios específico
    pN   = stVarN[8]

    y = stVar0[3] #Passando o parâmetro de estado
    qN = stVar0[7]
    pN = stVar0[8]
    tetaN = stVar0[10]
    MtetaN = stVar0[11]
    bulk = stVar0[15]
    shear = stVar0[16]
    

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

    for i in range (9):
            stressN_flat [i] = stress0_flat[i]

    #---------------------------------------------------------------------------
    #Correcao Plastica
    #---------------------------------------------------------------------------
    c = 0.0
    #Teste para pular correction
    if pN <= psmall: #pula o loop caso r_p seja menor que psmall
        yield1[0] = 0.0
        if ocr < 1.0: ocr =1.0
        if ocr == 1.0: p0 = psmall
        pN = psmall
        qN = 0.0
    else:
        while fabs(yield1[0]) > Ftol[0] and c <  maxits[0] and pN > psmall:
            c += 1.0
            '''
            if c > 1.0:
                printf("\nid\n")
                printf("%d\n",idx)
                printf("c\n")
                printf("%.16e\n",c)
                printf("yield1[0]\n")
                printf("%.16e\n",yield1[0])
            '''
            '''
            printf("idx\n")
            printf("%d\n",d_idx)
            printf("dT\n")
            printf("%.16e\n",dT_ar)
            printf("T\n")
            printf("%.16e\n",T)
            printf("residual\n")
            printf("%.16e\n",residual)
            '''

            #Calculando tensões principais iniciais
            eig_vals_3x3_analytical_erick(stressN_flat, eigvals)
            s1 = - eigvals[0] #compressão positivo
            s2 = - eigvals[1] 
            s3 = - eigvals[2]

            #Calculando os invariantes de cambrigde iniciais (0)
            p = (s1 + s2 + s3) / 3.0    
            q = sqrt(((s1 - s2)**2.0) + ((s2 - s3)**2.0) + 
                ((s3 - s1)**2.0)) / sqrt(2.0)
            teta = 0.0
            Mteta = 0.0
            if lode == 0.0: 
                teta = 30.0 #Mlode = Mtc
                Mteta = Mtc
            else:
                if fabs(s1 - s3) < psmall: 
                    teta = 30.0 #condição isotrópica
                    Mteta = Mtc  
                else:
                    teta = ((atan((s1 - 2.0 * s2 + s3) / 
                        (sqrt(3.0)*(s1 - s3)))) * 180.0 / pi)  
                    
                    Mteta = Mtc * ((2.0 * a)/(1.0 + a + (1.0 - a) * 
                        (sin(-3.0 * teta * pi / 180.0))))**(1.0 / 4.0)
            
            #Matriz elástica
            bulk = v * p / kappa
            if bulk < 10e3 * p1: bulk = 10e3 * p1 #limite inferior bulk=1 MPa, p1=1 kPa
            if bulk > 100e6 * p1: bulk = 100e6 * p1  #limite superior bulk=100 GPa
            shear = 1.5 * bulk * (1.0 - 2.0 * nu) / (1.0 + nu) 
            alpha1 = bulk + 4.0 * shear / 3.0
            alpha2 = bulk - 2.0 * shear / 3.0

            #Superfície de plastificação
            yield1[0] = ((q**n) / ((Mteta * p)**n) + 
                        log(p / p0) / log(r))

            #-------------------------------------------
            #Derivadas da superfície de plastificação
            #-------------------------------------------
       
            fa = 0.0 #df/ds1
            fb = 0.0 #df/ds2
            fc = 0.0 #df/ds3

            fax1 = (Mteta**n) * ((3.0 * p)**(n - 1.0))
            fax2 = (3.0**n) * log(r)
            fax3 = (n * log(p / p0)) + 1.0
            fax4 = n / 2.0
            fax5 = ((s1**2.0) + (s2**2.0) + (s3**2.0) -
                s1*s2 - s1*s3 - s2*s3)
            if fax5 < 0.0: fax5 = 0.0
            fax6 = fax5**((n - 2.0) / 2.0)

            fa = ((fax1 * fax3 / fax2) + (fax4 * fax6) *
                    (2.0 * s1 - s2 - s3))
            fb = ((fax1 * fax3 / fax2) + (fax4 * fax6) * 
                    (2.0 * s2 - s1 - s3)) 
            fc = ((fax1 * fax3 / fax2) + (fax4 * fax6) * 
                    (2.0 * s3 - s1 - s2)) 
        
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
                pc = ((p / (m - 1.0)) * (((q / (Mteta * p))**m) - 
                    1.0 + m))

                gax1 = (m * (Mteta**m) * ((3.0 * p)**(m - 2.0)))
                gax2 = 3.0**m
                gax3 = 3.0 * p * (m - 1.0)
                gax4 = 3.0 * pc * (2.0 - m - (1.0 / m))
                gax5 = m / 2.0
                gax6 = ((s1**2.0) + (s2**2.0) + (s3**2.0) -
                    s1*s2 - s1*s3 - s2*s3)
                if gax6 < 0.0: gax6 = 0.0
                gax7 = gax6**((m - 2.0) / 2.0)

                ga = (gax1 * (gax3 + gax4) / gax2 + (gax5 * gax7) * 
                    (2.0 * s1 - s2 - s3))
                gb = (gax1 * (gax3 + gax4) / gax2 + (gax5 * gax7) * 
                    (2.0 * s2 - s1 - s3))
                gc = (gax1 * (gax3 + gax4) / gax2 + (gax5 * gax7) * 
                    (2.0 * s3 - s1 - s2))
    
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

            lambdaS = 0.0
            if beta == 0.0:
                lambdaS = 0.0
            else: 
                lambdaS = ((Mteta * p)**n) * yield1[0] / beta    
                #if lambdaS < 0.0: lambdaS = 0.0
            
            s1_N = s1 - bGa * lambdaS
            s2_N = s2 - bGb * lambdaS
            s3_N = s3 - bGc * lambdaS

            pN = (s1_N + s2_N + s3_N) / 3.0    
            qN = sqrt(((s1_N - s2_N)**2.0) + ((s2_N - s3_N)**2.0) + 
                    ((s3_N - s1_N)**2.0)) / sqrt(2.0)

            #
            # Nova superfície de plastificação 
            yieldN = 1.0
            tetaN = teta
            MtetaN = Mteta #tenho dúvidas se isto esta certo, pois deveria manter o Mteta igual não?
            #p0N = p0

            if pN < psmall: #Controle de material superficial
                pN = psmall
                qN = 0.0 #inseri isso agora
                yieldN = 0.0 #inseri isso agora
            else: 
                yieldN = ((qN**n) / ((MtetaN * pN)**n) + log(pN / p0) / 
                        log(r))

            if fabs (yieldN) > fabs (yield1[0]):
                #Abandona a correção anterior
                lambdaS = (((Mteta * p)**n) * yield1[0] / 
                            (fa * fa + fb * fb + fc * fc))
                if lambdaS < 0.0: lambdaS = 0.0

                if isnan(lambdaS):
                    if prt == 1.0:
                        printf("\nERROlambdaS[0]_Correction:\n")
                        printf("%.16e\n",lambdaS)
                        printf("\nParticula:\n")
                        printf("%d\n",idx)
                    lambdaS = 0.0
                
                if isnan(bGa or bGb or bGc):
                    if prt == 1.0:
                        printf("\nERRObGa_Correction:\n")
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
                
                s1_N = s1 - bGa * lambdaS
                s2_N = s2 - bGb * lambdaS
                s3_N = s3 - bGc * lambdaS

                pN = (s1_N + s2_N + s3_N) / 3.0    
                qN = sqrt(((s1_N - s2_N)**2.0) + ((s2_N - s3_N)**2.0) + 
                    ((s3_N - s1_N)**2.0)) / sqrt(2.0)
                
                yieldN = ((qN**n) / ((MtetaN * pN)**n) + log(pN / p0) / 
                        log(r))
            
            if q == 0.0: #condição isotrópica
                correction = p - pN #correction compressão é positivo
                stressN_flat[0] += correction #stressN_flat compressão é negativo ( por isso soma corretion)
                stressN_flat[4] += correction #compressão positivo
                stressN_flat[8] += correction #compressão positivo   
            else:
                dVal = qN / q
                for i in range (9):
                    if i % 4 == 0:
                        stressN_flat[i] = (stressN_flat[i] + p) * dVal - pN
                    else:
                        stressN_flat [i] = stressN_flat [i] * dVal

            yield1[0] = yieldN
            '''
            if c > 2.0:
                printf("yield1[0]\n")
                printf("%.16e\n",yield1[0])
            '''
    
    if ocr > 1.0:
        p0aux = (pN * exp(log(r) * ((qN / (MtetaN * pN))**n)))
        ocr = p0 / p0aux
    
    y = v + _lambda * log(pN / p1) - gamma
    
    stVarN[0] = stVar0[0]
    stVarN[1] = ocr
    stVarN[2] = stVar0[2]
    stVarN[3] = y
    stVarN[4] = stVar0[4]
    stVarN[5] = stVar0[5]
    stVarN[6] = stVar0[6]
    stVarN[7] = qN
    stVarN[8] = pN
    stVarN[9] = qN / pN
    stVarN[10] = tetaN
    stVarN[11] = MtetaN
    stVarN[12] = qN / (pN * MtetaN)
    stVarN[14] = yield1[0]
    stVarN[15] = bulk
    stVarN[16] = shear
    stVarN[17] = stVar0[17]

    for i in range (18):
        if isnan(stVarN[i]):
            if prt == 1.0:
                printf("\nERRO_Correction_stVarN:\n")
                printf("%d\n",i)
                printf("%.16e\n",stVarN[i])
                printf("\nParticula:")
                printf("%d\n",idx)
            stVarN[i] = stVar0[i]
    
    for i in range (9):
        if isnan(stressN_flat[i]):
            if prt == 1.0:
                printf("\nERRO_Correction_stressN_flat:\n")
                printf("%d\n",i)
                printf("%.16e\n",stressN_flat[i])
                printf("\nParticula:")
                printf("%d\n",idx)
            stressN_flat[i] = stress0_flat[i]

    '''
    if fabs(yield1[0]) > Ftol[0]:
                printf("\nid\n")
                printf("%d\n",idx)
                printf("c\n")
                printf("%.16e\n",c)
                printf("yield1[0]\n")
                printf("%.16e\n",yield1[0])
    '''

                




            




