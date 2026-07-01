#Imports
from math import (pi, fabs, sqrt, log, pow, exp, tan, sin, cos, acos, isnan,
                  atan)

from matrix_operations import (eig_vals_3x3_analytical_erick,
                                matrix_multiply_vector, flat_to_voight,
                               voight_to_flat, matrix_transpose,
                               matrix_multiply, matrix_eigenvectors,
                               matrix_eigenvectors_erick, matrix_zero)

from compyle.api import declare

def pegasus_unload(props = [0], stress0_flat = [0], stVar0 = [0], eps = [0],
                Ftol = [0], maxits = [0], nsub = 4, pegasus = [0], 
                pegasus0 = [0], pegasus1 = [0]):
   

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
        dpegasus = (pegasus1[0] - pegasus0[0]) / nsub

        while c1 < nsub:
            c1 += 1.0
            c2 += 1.0
            pegasus[0] = pegasus0[0] + dpegasus 

            #Matriz elástica 
            evInc = - pegasus[0] * (eps[0] + eps[1] + eps[2])
            bulk = p_0 * (exp (v * evInc / kappa) - 1.0)/ evInc 
            if evInc == 0.0: bulk = v * p_0 / kappa 
            #if bulk < 1e6: bulk = 1e6 #limite inferior bulk=1 MPa
            #if bulk > 1e11: bulk = 1e11  #limite superior 100 GPa
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
                    c1 = nsub + 1.0 #sai do loop 2
                    c = maxits[0] + 1.0 #sai do loop 2
                else:
                    pegasus0[0] = 0.0
                    yield_0 = yield_save
                    c1 = nsub + 1.0 #sai do loop 2
            else:
                pegasus0[0] = pegasus[0]
                yield_0 = yieldN
 