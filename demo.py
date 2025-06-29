import torch
import cv2
import argparse
import utils.saveload
import utils.basic
import utils.improc
import PIL.Image
import numpy as np
import os
from prettytable import PrettyTable
import time

def read_mp4(name_path):
    vidcap = cv2.VideoCapture(name_path)
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    vidcap.release()
    return frames

def draw_pts(rgb, pts, visibs, confs, colors, radius=2, conf_thr=0.1, inds=None):
    H,W,C = rgb.shape
    assert(C==3)
    N,D = pts.shape
    assert(D==2)
    if inds is not None:
        pts = pts[inds]
        visibs = visibs[inds]
        confs = confs[inds]
        colors = colors[inds]
    for ii in range(N):
        xy = pts[ii].round().astype(np.int32)
        color = (int(colors[ii,0]),int(colors[ii,1]),int(colors[ii,2]))
        if visibs[ii] > 0.5:
            thickness = -1 # filled in
        else:
            thickness = 1 # hollow
        if confs[ii] > conf_thr:
            cv2.circle(rgb, (xy[0], xy[1]), radius, color, thickness)
    return rgb

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params

def forward_video(rgbs, model, args):
    
    B,T,C,H,W = rgbs.shape
    assert C == 3
    device = rgbs.device
    assert(B==1)

    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
    grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

    torch.cuda.empty_cache()
    print('starting forward...')
    f_start_time = time.time()

    flows_e, visconf_maps_e, _, _ = \
        model(rgbs[:, args.query_frame:], iters=args.inference_iters, sw=None, is_training=False)
    traj_maps_e = flows_e + grid_xy # B,Tf,2,H,W
    if args.query_frame > 0:
        backward_flows_e, backward_visconf_maps_e, _, _ = \
            model(rgbs[:, :args.query_frame+1].flip([1]), iters=args.inference_iters, sw=None, is_training=False)
        backward_traj_maps_e = backward_flows_e + grid_xy # B,Tb,2,H,W, reversed
        backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
        backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
        traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1) # B,T,2,H,W
        visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1) # B,T,2,H,W
    ftime = time.time()-f_start_time
    print('finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
    # traj_maps_e = flows_e + grid_xy # B,T,2,H,W
    utils.basic.print_stats('traj_maps_e', traj_maps_e)
    utils.basic.print_stats('visconf_maps_e', visconf_maps_e)

    # subsample to make the vis more readable
    rate = args.subsample_rate
    trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
    print('trajs_e', trajs_e.shape)
    xy0 = trajs_e[0,0].cpu().numpy()
    colors = utils.improc.get_2d_colors(xy0, H, W)

    # sort according to velocity, so that moving points are drawn last
    vels = trajs_e[0,1:].detach().cpu().numpy() - trajs_e[0,:-1].detach().cpu().numpy() # T-1,N,2
    vels = np.linalg.norm(vels, axis=-1).mean(axis=0)
    inds = np.argsort(vels)

    fn = args.mp4_path.split('/')[-1].split('.')[0]
    rgb_out_f = './pt_vis_%s_rate%d_q%d.mp4' % (fn, rate, args.query_frame)
    print('rgb_out_f', rgb_out_f)
    temp_dir = 'temp_pt_vis_%s_rate%d_q%d' % (fn, rate, args.query_frame)
    utils.basic.mkdir(temp_dir)
    vis = []
    for ti in range(T):
        pt_vis = draw_pts(rgbs[0,ti].permute(1,2,0).detach().cpu().byte().numpy().copy(),
                          trajs_e[0,ti].detach().cpu().numpy(),
                          visconfs_e[0,ti,:,0].detach().cpu().numpy(),
                          visconfs_e[0,ti,:,1].detach().cpu().numpy(),
                          colors=colors,
                          radius=max(int(rate//2),1),
                          inds=inds)
        vis.append(pt_vis)
    for ti in range(T):
        temp_out_f = '%s/%03d.png' % (temp_dir, ti)
        im = PIL.Image.fromarray(vis[ti])
        im.save(temp_out_f, "PNG", subsampling=0, quality=100)
    os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 30 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))

    # # flow vis
    # rgb_out_f = './flow_vis.mp4'
    # temp_dir = 'temp_flow_vis'
    # utils.basic.mkdir(temp_dir)
    # vis = []
    # for ti in range(T):
    #     flow_vis = utils.improc.flow2color(flows_e[0:1,ti])
    #     vis.append(flow_vis)
    # for ti in range(T):
    #     temp_out_f = '%s/%03d.png' % (temp_dir, ti)
    #     im = PIL.Image.fromarray(vis[ti][0].permute(1,2,0).cpu().numpy())
    #     im.save(temp_out_f, "PNG", subsampling=0, quality=100)
    # os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 24 -pattern_type glob -i "./%s/*.png" -c:v libx264 -crf 1 -pix_fmt yuv420p %s' % (temp_dir, rgb_out_f))
    
    return None

def run(model, args):
    log_dir = './logs_demo'
    
    global_step = 0

    if args.ckpt_init:
        _ = utils.saveload.load(
            None,
            args.ckpt_init,
            model,
            optimizer=None,
            scheduler=None,
            ignore_load=None,
            strict=True,
            verbose=False,
            weights_only=False,
        )
        print('loaded weights from', args.ckpt_init)
    else:
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        print('loaded weights from', url)

    model.cuda()
    for n, p in model.named_parameters():
        p.requires_grad = False
    model.eval()

    rgbs = read_mp4(args.mp4_path)
    print('rgbs[0]', rgbs[0].shape)
    H,W = rgbs[0].shape[:2]
    
    # shorten & shrink the video, in case the gpu is small
    rgbs = rgbs[:400]
    HH = 1024
    scale = min(HH/H, HH/W)
    H, W = int(H*scale), int(W*scale)
    H, W = H//8 * 8, W//8 * 8 # make it divisible by 8
    rgbs = [cv2.resize(rgb, dsize=(W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    print('rgbs[0]', rgbs[0].shape)

    # move to gpu
    rgbs = [torch.from_numpy(rgb).permute(2,0,1) for rgb in rgbs]
    rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).float() # 1,T,C,H,W
    print('rgbs', rgbs.shape)
    
    with torch.no_grad():
        metrics = forward_video(rgbs, model, args)
    
    return None

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_init", type=str, default='') # the ckpt we want (else default)
    parser.add_argument("--mp4_path", type=str, default='./demo_video/monkey.mp4') # input video 
    parser.add_argument("--query_frame", type=int, default=0) # which frame to track from
    parser.add_argument("--inference_iters", type=int, default=4) # number of inference steps per forward
    parser.add_argument("--window_len", type=int, default=16) # model hyperparam
    parser.add_argument("--subsample_rate", type=int, default=4) # vis hyp
    parser.add_argument("--mixed_precision", action='store_true', default=False)
    args = parser.parse_args()

    from nets.alltracker import Net; model = Net(args.window_len)
    count_parameters(model)

    run(model, args)
    
