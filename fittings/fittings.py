# -*- coding: cp1257 -*-
from numpy import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import t as tdist

'''
implementeeritud fittingud:
fit_single_shape - etteantud kuju lineaarne modifikatsioon
fit_twocomp - kahe etteantud kuju lin. kombinatsioon
fit_SGauss - "Skewed Gaussian" kuju
fit_Linear - Lineaarne fn
fit_DLorentz - 2 Lorentzi summa (nt. rubiini spekter)
fit_DLorentz_slope - eelmine koos kaldbäcki arvestamisega

'''




def analysePeaks(a,ax):
    #Kui meil on mingigi peaki sisaldav array koos vastava x-iarrayga, siis saada sealt kohe mingid raw andmed: y0, peakpos, FWHM, peakint
    #aga vaata, et ta ka 2D a-ga töötaks!
    #Teeks 1D spektri jaoks asja siiski lihtsamaks
    is1D = False
    if len(a.shape) == 1:
        a = array([a])
        is1D = True
    #Aga see on väga crude, nii et teeme lihtsalt
    y0 = amin(a, axis = 1)
    A = amax(a,axis = 1) - y0
    xc = ax[a.argmax(axis = 1)]
    #FWHMide määramine
    HMlevel = (A + y0) / 2
    xstep = (ax[len(ax) - 1] - ax[0]) / (len(ax) - 1)
    w = zeros(len(a))
    for i in range(len(a)):
        maxpos=a[i].argmax()
        rpart = where(a[i][maxpos:]<HMlevel[i])[0]
        width = rpart[0] if len(rpart) > 0 else len(a[i]) - maxpos - 1
        lpart = where(a[i][maxpos::-1]<HMlevel[i])[0]
        width += lpart[0] if len(lpart) > 0 else maxpos
        w[i] = width * xstep
    #ühe spektri korral returnime üheainsa tuple
    if is1D:
        return y0[0],xc[0],w[0],A[0]
    else:
        return y0,xc,w,A


def fit_general(a, ax, func, pinit, fitrange = (None, None), cyclic = True, conflevel = 0.99, errcoef = None):
    #fittimise üldmeetod; eri funktsioonidega fittimised kasutavad seda
    #annab tagasi fititud ja parameetrite maatriksi.
    #fitrange : kui on None, siis vastavad ax-le
    #cyclic: kas initsialiseerida fittingud tsükliliselt (järgmine eelmise tulemustega)
    
    #a peaks olema 2D, isegi kui ta sisaldab ainult üht spektrit
    if len(a.shape) == 1:
        a = a.reshape(1,*a.shape)
    #kuna me lõpptulemuse suurust teame (params puhul piniti järgi), siis oleks tegelikult siin hea dimensioneerida kohe õige suurusega array-d
    #ja initsialiseerida nad 1-deks, sest null võib arvutustel sageli halvasti käituda (vaja temaga jagada vms.)
    fitted = ones(a.shape) #siin on tegelikult küsimus, kas völjastada fitting ainult fitrange osas või kogu osas. Ja kas noissi arvutada kus.
    params = ones((a.shape[0], len(pinit) * 2 + 2)) #kõik param + vead + noise + s/n
    
    #fittimisvahemik
    p = [0,a.shape[1] - 1]
    for k in range(2):
        if fitrange[k] is not None:
            p[k] = int((fitrange[k] - ax[0]) / (ax[1] - ax[0]))

    df = p[1] - p[0] - len(pinit) #vabadusastmete arv
    if errcoef is None: #Studenti koefitsient vastavalt  etteantud konfidentsile:
        errcoef = tdist.interval(conflevel, df, loc=0, scale=1)[1]
    
    #peatsükkel
    for i in range(a.shape[0]):
        #diagnostika: saadab tagasi alglähenduse
        #fitted[i] = func(ax, *pinit)
        #continue

        try:
            popt, pcov = curve_fit(func, ax[p[0]:p[1]], a[i,p[0]:p[1]],tuple(pinit)) 
        except RuntimeError: #siis järelikult ei taha koonduda
            #vastavad read jäävad täidetuks ühtedega
            continue
        
        try:
            perr = [errcoef * math.sqrt(pcov[n,n]) for n in range(len(popt))]
        except TypeError: #mingil juhul seal pcov = inf ja käitub floadina
            #print i,popt,pcov
            continue
        except ValueError:
            continue
        fitted[i] = func(ax,*popt)
        #tsükliline initsieerimine
        if cyclic:
            pinit = popt
        #rmse ja müra hinnangud
        errvect = (a[i] - fitted[i])**2
        noise = sqrt(errvect.sum() / a.shape[1]) #siin on chi-ruudu saamine, kuidas tuleks?
        signal = max(fitted[i]) - min(fitted[i])
        #signal/noise tuleks siit kuhugi ära vist viia, sest pole väga üldine
        #lõpuks siis ka parameetrite rida
        params[i] = array([perr[n//2] if n%2 else popt[n//2] for n in range(len(popt) * 2)] + [noise] + [signal / noise])
    return fitted,params


def fit_single_shape_colnames():
    return ["a","a-Err", "b", "b-Err", "Noise", "S/N"]

def fit_single_shape(a, ax, shp, fitrange = (None, None), cyclic = True):

    if len(a.shape) == 1:
        a = a.reshape(1,*a.shape)
    
    f = interp1d(*shp)
    def func(x,a,b):
        return a*f(x) + b
    pinit = [max(a[0]) / max(shp[1]), 0] #pärameetrite init
    return fit_general(a, ax, func, pinit, fitrange, cyclic = cyclic) #siia võib lisada kas errcoefi või confleveli kaa 
    #(returnib fitted, params)


def fit_twocomp_colnames():
    return ["ALH1","ALH1-Err","DLH1","DLH1-Err","ALH2","ALH2-Err","DLH2","DLH2-Err","Y0","Y0-Err", "Noise", "S/N", "Asum", "Asum-Err", "LH1pos","LH2pos","LH2/LH1","Ratio-Err"]

def fit_twocomp(a, ax, comp1, comp2, fitrange = (None, None),  max1 = 889.1, max2 = 858.6, cyclic = True):
    #print 'fitib a kahekomponentselt'
    #(äkki võiks mõelda ka fiti komponentide tagastamisele?)
    #proovime siis nii, et comp1,2 on 2realised array-d, üks rida lainepikkusi ja teine spekter, aga fititavatel (a) on eraldi 1D x array (ax)
    
    #interpolatsioonifn-d
    x1=comp1[0]
    y1=comp1[1]
    f1=interp1d(x1,y1)
    x2=comp2[0]
    y2=comp2[1]
    f2=interp1d(x2,y2)
    
    #siin oleks vaja mingit kontrolli, et interpol. piiridest välja ei lähe
    def func(x, a1, d1, a2, d2, y0):
        #print a1,d1,a2,d2,y0
        return a1 * f1(x - d1) + a2 * f2(x - d2) + y0
    
    if len(a.shape) == 1:
        a = a.reshape(1,*a.shape)
    est = sum(a[0]) /100
    pinit = [est,0.0,est,0.0,0.0]
    fitted,params = fit_general(a, ax, func, pinit, fitrange, cyclic = cyclic)
    #lisame arvutatud ridu: Asum, Asum-Err, LH1pos, LH2pos, LH2/LH1, Ratio-Err
    params = params.T
    params = vstack((params, params[0] + params[4], params[1] + params[5], params[2] + max1, params[6] + max2, params[4]/params[0], (params[1]/params[0] + params[5]/params[6]) * params[6] / params[0]))
    params = params.T
    
    return fitted, params

def fit_SGauss_colnames():
    return ["xc","xc-Err","w","w-Err","A","A-Err","b","b-Err","y0","y0-Err","Noise","S/N"]

def fit_SGauss(a, ax, fitrange = (None, None), max = 850.0, width = 40.0, amplitude = 10.0, skewness = 0.3, y0 = 0.0, cyclic = True):
    
    def func(x, xc, w, A, b, y0):
        ut = zeros(x.size) + y0
        c = 1 + 2*b*(x-xc)/w
        ut[c>0] += A*exp(-log(2)*(log(c[c>0])/b)**2)
        return ut
    
    fitted,params = fit_general(a, ax, func, [max,width,amplitude,skewness,y0], fitrange, cyclic)
    return fitted,params
    
def fit_Linear(a,ax,fitrange = (None, None), cyclic = True):
    #initsialiseerime parameetrid kahe esimese punkti järgi
    A = a[1,0] - ax[1] * (a[1,0] - a[0,0]) / (ax[1] - ax[0])
    B = (a[1,0] - a[0,0]) / (ax[1] - ax[0])
    
    def func(x, A, B):
        return A + B * x
        
    fitted,params = fit_general(a, ax, func, [A, B], fitrange, cyclic)
    return fitted,params
    
def fit_DLorentz(a, ax, divisionx = None, paramlist = None):
    #proovime algvrt. leida a[0]-st
    #kui paramlist [xc1, w1, A1, xc2, w2, A2, y0] on antud, proovitakse kasutada
    #muidu proovitakse leida lähendväärtusi automaatselt
    if len(a.shape) == 1:
        a = a.reshape(1,*a.shape)
    if (divisionx is None):
        divisionx = ax[int(len(ax)/2)] #jagame lihtsalt pooleks
    if (paramlist is None):
        divi = len(ax[ax < divisionx])
        y0,xc1,w1,A1 = analysePeaks(a[0][:divi],ax)
        y0,xc2,w2,A2 = analysePeaks(a[0][divi:],ax[divi:])
        paramlist = [xc1, w1, A1, xc2, w2, A2, y0]
    
    def func(x, xc1, w1, A1, xc2, w2, A2, y0):
        ut = zeros(x.size) + y0
        ut += A1 / (w1 + ((x-xc1)**2 / w1))
        ut += A2 / (w2 + ((x-xc2)**2 / w2))
        return ut
    
    #fitrange võib siin tahta täpsustamist (vst xc ja w hinnangutele jne)
    fitted,params = fit_general(a, ax, func, paramlist, (None, None), True) #cyclic?
    return fitted,params

def fit_DLorentz_colnames():
    return ["xc1","xc1-Err","w1","w1-Err","A1", "A1-Err","xc2", "xc2-Err", "w2","w2-Err", "A2", "A2-Err", "y0", "y0-Err", "Noise", "S/N"]

def fit_DLorentz_slope(a, ax, divisionx = None, paramlist = None):
    #proovime algvrt. leida a[0]-st
    #kui paramlist [xc1, w1, A1, xc2, w2, A2, y0, k] on antud, proovitakse kasutada
    #muidu proovitakse leida lähendväärtusi automaatselt
    if len(a.shape) == 1:
        a = a.reshape(1,*a.shape)
    if (divisionx is None):
        divisionx = ax[int(len(ax)/2)] #jagame lihtsalt pooleks
    if (paramlist is None):
        divi = len(ax[ax < divisionx])
        y0,xc1,w1,A1 = analysePeaks(a[0][:divi],ax)
        y0,xc2,w2,A2 = analysePeaks(a[0][divi:],ax[divi:])
        
        k = (a[0][-1]-a[0][0]) / (ax[-1]-ax[0])
        paramlist = [xc1, w1, A1, xc2, w2, A2, y0, k]
    xc = (ax[-1]+ ax[0])/2

    def func(x, xc1, w1, A1, xc2, w2, A2, y0, k):
        ut = zeros(x.size) + y0
        ut += A1 / (w1 + ((x-xc1)**2 / w1))
        ut += A2 / (w2 + ((x-xc2)**2 / w2))
        ut += k*(x-xc) 
        return ut
    
    #fitrange võib siin tahta täpsustamist (vst xc ja w hinnangutele jne)
    fitted,params = fit_general(a, ax, func, paramlist, (None, None), True) #cyclic?
    return fitted,params

def fit_DLorentz_slope_colnames():
    return ["xc1","xc1-Err","w1","w1-Err","A1", "A1-Err","xc2", "xc2-Err", "w2","w2-Err", "A2", "A2-Err", "y0", "y0-Err", "k", "k-Err", "Noise", "S/N"]
