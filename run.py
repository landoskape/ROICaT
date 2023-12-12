from argparse import ArgumentParser
from vrAnalysis import database
from vrAnalysis import fileManagement
from vrAnalysis import session
from roicat.pipelines import pipeline_tracking
from roicat.util import get_default_parameters
from torch import tensor

WORKFLOWS = ['tracking']

sessiondb = database.vrDatabase('vrSessions')
mousedb = database.vrDatabase('vrMice')

# === type definitions for argument parser ===
def workflow_type(string):
    """duck-type string to check if it is a valid ROICaT pipeline"""
    try:
        get_default_parameters(pipeline=string)
    except:
        raise ValueError(f"workflow {string} not supported")
    else:
        return string

# === argument parser for run ===    
def handle_arguments():
    parser = ArgumentParser(description='Arguments for running an ROICaT pipeline.')
    parser.add_argument('--mouse', type=str, required=True, help='the name of the mouse to process')
    parser.add_argument('--session', type=str, nargs=2, required=False, help='if used, requires date and sessionid string, and performs ROICaT on single session')
    parser.add_argument('--workflow', type=workflow_type, required=True, help='which workflow to perform')
    parser.add_argument('--no-database-update', default=False, action='store_true', help='if used, will not do a database update')
    parser.add_argument('--nosave', action='store_true', default=False, help='whether to prevent saving')
    return parser.parse_args()

# === custom method for getting the appropriate dir_outer when tracking across sessions ===
def define_dirs_across(args):
    # first define save path and save name
    dir_save = fileManagement.localDataPath() / args.mouse
    name_save = lambda planeName: args.mouse + '.' + planeName
    
    # identify sessions for requested mouse (that match relevant criteria in database)
    vrdb = database.vrDatabase('vrSessions')
    ises = vrdb.iterSessions(imaging=True, mouseName=args.mouse, dontTrack=False)
    assert len(ises)>0, f"no sessions found for mouse={args.mouse}"
    session_paths = [ses.sessionPath() for ses in ises]

    # for identified sessions, determine which planes are present
    plane_in_ses = [set([pn.stem for pn in ses.suite2pPath().rglob('plane*')]) for ses in ises]
    plane_names = set.union(*plane_in_ses)
    assert all([planes == plane_names for planes in plane_in_ses]), "requested sessions have different sets of planes"
    from natsort import natsorted
    plane_names = natsorted(list(plane_names), key=lambda x: x.lower()) 

    # inform the user what was found for this run
    print('')
    print(f"Running ROICaT:{args.workflow} on the following sessions:")
    for idx, ises in enumerate(ises):
        print('  ', idx, ises.sessionPrint())
    
    print('')
    print(f"Using plane_names: {', '.join(plane_names)}")

    # return to main
    return session_paths, plane_names, dir_save, name_save


def define_dirs_within(args, ses):
    """custom method for getting the appropriate dir_outer when tracking within session"""

    # first define save path and save name
    dir_save = ses.sessionPath()
    name_save = args.mouse + '.' + 'within_session'
    
    plane_paths = [ses.suite2pPath() / plane_name for plane_name in ses.value['planeNames']]

    # inform the user what was found for this run
    print('')
    print(f"Running ROICaT:{args.workflow} within session {ses.sessionPrint()} across the following planes:")
    for plane_name in ses.value['planeNames']: print(plane_name)

    # return to main
    return plane_paths, dir_save, name_save



def main():
    """main program that runs the requested pipeline"""
    
    # parse arguments
    args = handle_arguments()
    
    # define pipeline type (includes variations apart from args.workflow)
    if args.workflow=='tracking':
        pipeline_type = 'tracking_across' if args.session is None else 'tracking_within'
    else:
        raise ValueError(f"did not recognize workflow ({args.workflow})")

    # get default parameters for workflow    
    params = get_default_parameters(args.workflow) # get default params
    if pipeline_type == 'tracking_across':
        target_paths, plane_names, dir_save, name_save = define_dirs_across(args) # get session paths and planes to track
    
    elif pipeline_type == 'tracking_within':
        ses = session.vrExperiment(args.mouse, args.session[0], args.session[1]) # load requested session for tracking
        target_paths, dir_save, name_save = define_dirs_within(args, ses) # get plane paths for tracking within session

    # prepare paths in params dictionary
    params['data_loading']['dir_outer'] = target_paths # load target paths into the params dictionary
    if args.nosave: 
        params['results_saving']['dir_save'] = None
    else:
        params['results_saving']['dir_save'] = dir_save # indicate where to save results

    # run the requested pipeline
    if pipeline_type == 'tracking_across':
        # For tracking across sessions, perform tracking pipeline on each plane independently
        for _, plane_name in enumerate(plane_names):
            params['data_loading']['reMatch_in_path'] = plane_name # update planeName to filter paths by
            params['results_saving']['prefix_name_save'] = name_save(plane_name) # define what the name is (combination of mouse name and plane name)
            pipeline = pipeline_tracking(params) # create pipeline object
            _, _ = pipeline.run() # run the pipeline

    elif pipeline_type == 'tracking_within':
        params['results_saving']['prefix_name_save'] = name_save # define header of file name for saving
        roiActivity = ses.loadone('mpci.roiActivityF').T
        pipeline = pipeline_tracking(params, feature_additional=[tensor(roiActivity)], additional_feature_name=['fluorescence']) # create pipeline object
        _, _ = pipeline.run() # run the pipeline

    else:
        raise ValueError(f"did not recognize workflow ({args.workflow})")
    

if __name__ == "__main__":
    main()
