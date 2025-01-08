from pathlib import Path
import tempfile
import numpy as np
import roicat
from vrAnalysis import database


def get_mice():
    mousedb = database.vrDatabase("vrMice")
    tracked_mice = list(mousedb.getTable(tracked=True)["mouseName"])
    ignore_mice = []
    use_mice = [mouse for mouse in tracked_mice if mouse not in ignore_mice]
    return use_mice


def get_sessions(mouse_name):
    vrdb = database.vrDatabase("vrSessions")
    ises = vrdb.iterSessions(imaging=True, mouseName=mouse_name, dontTrack=False)
    pathList = []
    for ses in ises:
        pathList.append(ses.sessionPath())
    return pathList


def get_stat_ops_path(session_path):
    pathSuffixToStat = "stat.npy"
    pathSuffixToOps = "ops.npy"

    paths_stat = roicat.helpers.find_paths(
        dir_outer=session_path,
        reMatch=pathSuffixToStat,
        depth=6,
    )[:]
    paths_ops = np.array([Path(path).resolve().parent / pathSuffixToOps for path in paths_stat])[:]
    return paths_stat, paths_ops


def roicat_savepath(session_path):
    savepath = Path(session_path) / "roicat"
    savepath.mkdir(exist_ok=True)
    return savepath


if __name__ == "__main__":
    use_mice = get_mice()
    for mouse_name in use_mice:
        pathList = get_sessions(mouse_name)
        for session_path in pathList:
            print(f"Working on session: {session_path}")
            paths_stat, paths_ops = get_stat_ops_path(session_path)
            data = roicat.data_importing.Data_suite2p(
                paths_statFiles=paths_stat,
                paths_opsFiles=paths_ops,
                um_per_pixel=1.6,
                new_or_old_suite2p="new",
                type_meanImg="meanImgE",
                verbose=True,
            )
            if not data.check_completeness(verbose=False)["classification_inference"]:
                print(f"Data object is missing attributes necessary for tracking.")
                continue

            DEVICE = roicat.helpers.set_device(use_GPU=True, verbose=True)
            dir_temp = tempfile.gettempdir()

            roinet = roicat.ROInet.ROInet_embedder(
                device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
                dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
                download_method="check_local_first",  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
                download_url="https://osf.io/c8m3b/download",  ## URL of the model
                download_hash="357a8d9b630ec79f3e015d0056a4c2d5",  ## Hash of the model file
                forward_pass_version="head",  ## How the data is passed through the network
                verbose=True,  ## Whether to print updates
            )

            roinet.generate_dataloader(
                ROI_images=data.ROI_images,  ## Input images of ROIs
                um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
                pref_plot=False,  ## Whether or not to plot the ROI sizes
            )
            images = roinet.ROI_images_rs

            roinet.generate_latents()

            np.save(roicat_savepath(session_path) / "roinet_latents", roinet.latents)
            print(f"Saved latents to {roicat_savepath(session_path) / 'roinet_latents.npy'}")
