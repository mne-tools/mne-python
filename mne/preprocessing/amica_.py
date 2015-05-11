from collections import OrderedDict

import numpy as np
from scipy.special import gamma

from subprocess import call


def write_paramfile(raw_data_file,
                    paramdir,
                    paramfile,
                    do_reject = 1, 
                    numrej = 5, 
                    max_iter = 1000, 
                    max_threads = 3, 
                    dble_data = 0,
                    num_mix_comps = 1,
                    minlrate = 1.000000e-08,
                    use_min_dll = 0,
                    numprocs = 1,
                    doPCA = 0,
                    PCAkeep = None,
                    chans = None,
                    length = None):

    if not PCAkeep:
        PCAkeep = chans

    am_dict=OrderedDict()
    
    am_dict['files'] = raw_data_file
    am_dict['data_dim'] = chans
    am_dict['outdir'] = paramdir
    am_dict['field_dim'] = length
    am_dict['dble_data'] = dble_data
#    am_dict['byte_size'] = 4
    am_dict['load_rej'] = 0
    am_dict['load_A'] = 0
    am_dict['load_W'] = 0
    am_dict['load_c'] = 0
    am_dict['load_gm'] = 0
    am_dict['load_alpha'] = 0
    am_dict['load_mu'] = 0
    am_dict['load_beta'] = 0
    am_dict['load_rho'] = 0
    am_dict['load_comp_list'] = 0
    am_dict['num_models'] = 1
    am_dict['num_mix_comps'] = num_mix_comps
    am_dict['mineig'] = 1e-15
    am_dict['do_approx_sphere'] = 1
    am_dict['pdftype'] = 0
    am_dict['mineig'] = 1e-15
    am_dict['pcakeep'] = PCAkeep
    am_dict['share_comps'] = 0
    am_dict['share_start'] = 100
    am_dict['comp_thresh'] = '0.990000'
    am_dict['share_iter'] = 100
    am_dict['do_reject'] = do_reject
    am_dict['numrej'] = numrej
    am_dict['rejstart'] = 2
    am_dict['rejint'] = 3
    am_dict['rejsig'] = '3.00000'
    am_dict['do_newton'] = 1
    am_dict['newt_start'] = 50
    am_dict['lrate'] = '0.050000'
    am_dict['max_iter'] = max_iter
    am_dict['do_history'] = 0
    am_dict['histstep'] = 10
    am_dict['use_grad_norm'] = 1
    am_dict['use_min_dll'] = use_min_dll
    am_dict['min_grad_norm'] = '1.00000e-06'
    am_dict['min_dll'] = '1.00000e-06'
    am_dict['max_threads'] = max_threads
    am_dict['numprocs'] = numprocs
    am_dict['do_opt_block'] = 0
    am_dict['block_size'] = 256
    am_dict['fix_init'] = 0
    am_dict['print_debug'] = 0
    am_dict['num_samples'] = 1
    am_dict['field_blocksize'] = 1
    am_dict['minlrate'] = minlrate
    am_dict['lratefact'] = '0.500000'
    am_dict['rholrate'] = '0.050000'
    am_dict['rho0'] = '1.500000'
    am_dict['minrho'] = '1.000000'
    am_dict['maxrho'] = '2.000000'
    am_dict['rholratefact'] = '0.500000'
    am_dict['kurt_start'] = 3
    am_dict['num_kurt'] = 5
    am_dict['kurt_int'] = 1
    am_dict['newt_ramp'] = 10
    am_dict['newtrate'] = '1.000000'
    am_dict['writestep'] = 10
    am_dict['write_nd'] = 0
    am_dict['write_LLt'] = 1
    am_dict['decwindow'] = 1
    am_dict['max_decs'] = 3
    am_dict['update_A'] = 1
    am_dict['update_c'] = 1
    am_dict['update_gm'] = 1
    am_dict['update_alpha'] = 1
    am_dict['update_mu'] = 1
    am_dict['update_beta'] = 1
    am_dict['do_rho'] = 1
    am_dict['invsigmax'] = '100.000000'
    am_dict['invsigmin'] = '1.00000e-08' # 0
    am_dict['do_mean'] = 1
    am_dict['do_sphere'] = 1
    am_dict['doPCA'] = doPCA
#    am_dict['pcadb'] = '30.000000'
    am_dict['doscaling'] = 1
    am_dict['scalestep'] = 1
    
    outlines = []
    for key, value in zip(am_dict.keys(),am_dict.values()):
        outlines.append(key + " " + str(value))
    
    with open(paramfile, "w+") as f:
        f.write('\n'.join(outlines))


def loadmodout_convert(targetdir):

    try:
        with open(targetdir + '/gm', 'rb') as f:
            gm = np.fromfile(f,dtype=np.double)
        num_models = len(gm);
    except:
        print('1 model')
        num_models = 1
        gm = 1

    try:
        with open(targetdir + '/W', 'rb') as f:
            W = np.fromfile(f,dtype=np.double)
            nw2 = len(W)/num_models
            nw = int(np.sqrt(nw2))
            W = W.reshape((nw,nw,num_models), order = "FORTRAN")
    except:
       print('no valid AMICA decomposition found')

    num_pcs = nw

    try:
        with open(targetdir + '/mean', 'rb') as f:
            mn = np.fromfile(f,dtype=np.double)
            nx = len(mn)
            mnset = 1
            mxset = 1
    except:
       print('default mn')    
       mnset = 0
       nxset = 0

    try:
        with open(targetdir + '/S', 'rb') as f:
            S = np.fromfile(f,dtype=np.double)
            nx = int(np.sqrt(len(S)))
            S = S.reshape((nx,nx), order = "FORTRAN")
    except:
        if nxset:
            print('no sphering, default assumed')
            S = np.eye(nx)
        print('no sphering matrix or mn found!')

    if not mnset:
        mn = np.zeros(nx)

    data_dim = nx
    data_men = mn

    try:
        with open(targetdir + '/comp_list', 'rb') as f:
            comp_list = np.fromfile(f,dtype=np.int32)
            comp_list = comp_list[:num_models*nw]
            comp_list = comp_list.reshape((nw,num_models))
            complistset = 1
    except:
        complistset = 0

    try:
        with open(targetdir + '/alpha', 'rb') as f:
            alphatmp = np.fromfile(f,dtype=np.double)
            num_mix = int(len(alphatmp)/(nw*num_models))
            alpha = np.zeros((nw,num_mix,num_models))
            alphatmp = alphatmp.reshape(nw*num_models,num_mix)
            for h in range(0,num_models):
                for i in range(0,nw):
                    alpha[i,:,h] = alphatmp[comp_list[i,h]-1,:]
    except:
        print('default alpha')            
        num_mix = 1;
        alpha = np.ones((num_mix,nw,num_models));

    num_mix_used = (np.ones((nw,num_models))*sum(alpha[0,:,0,]>0)).astype('int')

    mu = np.zeros((num_mix,nw,num_models))
    try:
        with open(targetdir + '/mu', 'rb') as f:
            mutmp = np.fromfile(f,dtype=np.double)
            mutmp = mutmp.reshape((nw*num_models,num_mix))
            for h in range(0,num_models):
                for i in range(0,nw):
                    mu[:,i,h] = mutmp[comp_list[i,h]-1,:]
    except:
        print('default mu')

    sbeta = np.ones((num_mix,nw,num_models))
    try:
        with open(targetdir + '/sbeta', 'rb') as f:
            sbetatmp = np.fromfile(f,dtype=np.double)
            sbetatmp = sbetatmp.reshape((nw*num_models,num_mix))
            for h in range(0,num_models):
                for i in range(0,nw):
                    sbeta[:,i,h] = sbetatmp[comp_list[i,h]-1,:]
    except:
        print('default sbeta')


    rho = np.ones((num_mix,nw,num_models))*2
    try:
        with open(targetdir + '/rho', 'rb') as f:
            rhotmp = np.fromfile(f,dtype=np.double)
            rhotmp = rhotmp.reshape((nw*num_models,num_mix))
            for h in range(0,num_models):
                for i in range(0,nw):
                    rho[:,i,h] = rhotmp[comp_list[i,h]-1,:]
    except:
        print('default rho')

    num_mod = len(gm);
    gmord = int(np.sort(gm,1)); # not sure if this would work with num_mod > 1
    W = W[:,:,gmord-1]
    alpha = alpha[:,:,gmord-1];
    mu = mu[:,:,gmord-1];
    sbeta = sbeta[:,:,gmord-1];
    rho = rho[:,:,gmord-1];

    if complistset:
        if num_mod > 1:
            comp_list = comp_list[gmord,:];

    w = np.asmatrix(W)
    s = np.asmatrix(S)
    if num_mod > 1:
        A = np.asmatrix(np.zeros((nx,nw)))
        for h in range(0,num_mod):
            A[:,:,h] = np.linalg.pinv(w[:,:,h]*s[:,:nw])
    else:
        A = np.linalg.pinv(w[:,:]*s[:nw,:])

    # the following doesn't yet work for h > 1
    svar = np.zeros(nw)
    for i in range(0,nw):
        sqmu = np.square(mu[:num_mix_used[i],i])
        gamrho = ((gamma(3 / rho[:num_mix_used[i],i] ) 
                 / gamma(1 / rho[:num_mix_used[i],i] )
                 / (np.square(sbeta[:num_mix_used[i],i])).T))
        svar[i] = sum(  alpha[i,:num_mix_used[i]] * (gamrho+sqmu).T )
        svar[i] = svar[i] * np.square(np.linalg.norm(A[:,i],ord=2))
    origord = ((np.argsort(svar))[::-1]).astype('int')

    outA = A[:,origord].copy()
    outW = w[origord,:].copy()

    for i in range(0,nw):
        na = np.linalg.norm(outA[:,i],ord=2);
        outA[:,i] = outA[:,i] / na
        outW[i,:] = outW[i,:] * na

    return outW, outA, S


def mne_amica(data,
              max_iter = 1000, 
            filter_low = 1, 
           filter_high = 40,
           max_threads = 1,
          numprocs = 1,
           outfile = None,
           amica_binary = None,
           targetdir = '/tmp/',
           max_pca_components = None):

    if not amica_binary:
        print("Please specify the full location of the amica binary!")

    outfile = targetdir + "amtemp"

    doPCA = 0
    if max_pca_components:
        doPCA = 1

#    n_rank = np.linalg.matrix_rank(data)
#    if pcakeep > n_rank:
#       print('''Non-fatal warning: 
#    data is rank deficient (channel interpolated? reference included? comps removed?),
#    consider using doPCA = 1 and pcakeep < less than the number of chans''')

    d = np.asarray(data)*1000000
    c = d.astype('<f32', copy=False)
    c.tofile(outfile)

    paramfile = targetdir + 'paramfile'

#    print(paramfile)
    write_paramfile(outfile, targetdir, paramfile, 
        chans = data.shape[1], length = data.shape[0],
        max_iter=max_iter, max_threads = max_threads, numprocs = numprocs, 
        doPCA=doPCA, PCAkeep=max_pca_components,
        num_mix_comps = 1)
    call([amica_binary + ' ' + paramfile], shell=True)
    
    W, A, S = loadmodout_convert(targetdir)
    
    # W: unmixing weights (post-sphering)
    # S: sphering matrix
    # A: model component matrices    
    return W, A, S

