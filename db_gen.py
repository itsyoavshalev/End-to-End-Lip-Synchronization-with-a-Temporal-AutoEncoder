from optparse import OptionParser
import time
import sys
import os
from scipy.io import wavfile
import python_speech_features
import cv2
import shutil
from numpy import genfromtxt
from common import *
import multiprocessing
from multiprocessing import Pool
import pims

parser = OptionParser()
parser.add_option('--config', type=str, help="db configuration", default="dbs_config.yaml")


def assert_video_length(p1, p2):
    video_1_frames = pims.Video(p1)
    video_2_frames = pims.Video(p2)
    assert len(video_1_frames) == len(video_2_frames)


def process_audio_file_wrapper(args):
    return process_audio_file(*args)


def process_audio_file(index, item, total, db_config, gen_db_path, raw_db_path, target_audio_rate, target_video_fps):
    start_time = time.time()

    folder_name, file_name = item.split('/')
    raw_folder_path = '{}{}/'.format(raw_db_path, folder_name)
    gen_folder_path = '{}{}/'.format(gen_db_path, folder_name)
    raw_video_file_path = '{}{}.mp4'.format(raw_folder_path, file_name)
    # temp_video_path = '{}tmp_{}.mp4'.format(gen_folder_path, file_name)
    converted_video_file_path = '{}con_{}.mp4'.format(gen_folder_path, file_name)
    converted_audio_file_path = '{}con_{}.wav'.format(gen_folder_path, file_name)

    command = 'ffmpeg -loglevel error -y -hwaccel cuvid -r {1} -i {0} -vf scale=320:320 -qscale:v 4 -async 1 -r {1} -deinterlace -ar {2} {3}'.format(
        raw_video_file_path, target_video_fps, target_audio_rate, converted_video_file_path)

    error = os.system(command)

    if error:
        msg = 'error while converting video'
        print(msg)
        raise Exception(msg)

    # assert_video_length(converted_video_file_path, raw_video_file_path)

    # clip = mp.VideoFileClip(temp_video_path, audio_fps=target_audio_rate)  # .subclip(0,20)
    # clip_resized = clip.resize(width=480, height=480)
    # clip_resized.write_videofile(converted_video_file_path, audio_fps=target_audio_rate)

    command = 'ffmpeg -loglevel error -y -hwaccel cuvid -i {} -ac 1 -acodec pcm_s16le -ar {} {}'.format(
        converted_video_file_path,
        target_audio_rate,
        converted_audio_file_path)
    error = os.system(command)

    if error:
        msg = 'error while converting video to audio'
        print(msg)
        raise Exception(msg)

    sample_rate, audio = wavfile.read(converted_audio_file_path)
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc]).astype(float)

    video_cap = cv2.VideoCapture(converted_video_file_path)
    video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # if (float(len(audio)) / target_audio_rate) < (float(video_len) / target_video_fps):
    #     raise Exception('audio is shorter then video')

    audio_features_path = '{0}{1}_af'.format(gen_folder_path, file_name)
    np.save(audio_features_path, mfcc)

    num_of_audio_frames_path = '{0}{1}_{2}'.format(gen_folder_path, file_name,
                                                   db_config.num_of_audio_frames_file_name)

    np.save(num_of_audio_frames_path, mfcc.shape[1])

    total_time = time.time() - start_time

    print('{} / {} {} done in {} seconds'.format(index, total, raw_video_file_path, total_time))


def assert_video_length_c(p1, count):
    video_1_frames = pims.Video(p1)
    assert len(video_1_frames) == count


def create_audio_features(db_config, gen_db_path, raw_db_path, raw_db_list_path, target_audio_rate, target_video_fps):
    start_audio_time = time.time()

    print('####### Creating audio features #######')

    if not os.path.isdir(raw_db_path):
        raise Exception('raw data folder does not exists')

    if not os.path.isdir(gen_db_path):
        os.makedirs(gen_db_path)

    audio_rate_path = '{}{}'.format(gen_db_path, db_config.audio_rate_file_name)
    np.save(audio_rate_path, target_audio_rate)

    with open(raw_db_list_path) as f:
        files_list = f.readlines()

    files_list = [x.strip() for x in files_list]

    for item in files_list:
        folder_name, file_name = item.split('/')
        gen_folder_path = '{}{}/'.format(gen_db_path, folder_name)

        if not os.path.exists(gen_folder_path):
            os.mkdir(gen_folder_path)

    num_cores = multiprocessing.cpu_count()

    with Pool(num_cores) as pool:
        params_p = [(index, item, len(files_list), db_config, gen_db_path, raw_db_path, target_audio_rate, target_video_fps) for index, item in
                    enumerate(files_list)]

        # process_audio_file(files_list[0], db_config, gen_db_path, raw_db_path, target_audio_rate, target_video_fps)

        pool.map(process_audio_file_wrapper, params_p)

    total_audio_time = time.time() - start_audio_time

    print('####### Audio features created in {} seconds #######'.format(total_audio_time))


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth_angles(all_frames_angles):
    box = 20

    all_frames_angles_x = all_frames_angles[:, 0]
    all_frames_angles_y = all_frames_angles[:, 1]
    all_frames_angles_z = all_frames_angles[:, 2]

    x = np.concatenate((np.repeat(all_frames_angles_x[0], box - 1), all_frames_angles_x[:], np.repeat(
        all_frames_angles_x[-1], box - 1)))
    x = smooth(x, box)[box:(box + len(all_frames_angles_x))]

    y = np.concatenate((np.repeat(all_frames_angles_y[0], box - 1), all_frames_angles_y[:], np.repeat(
        all_frames_angles_y[-1], box - 1)))
    y = smooth(y, box)[box:(box + len(all_frames_angles_y))]

    z = np.concatenate((np.repeat(all_frames_angles_z[0], box - 1), all_frames_angles_z[:], np.repeat(
        all_frames_angles_z[-1], box - 1)))
    z = smooth(z, box)[box:(box + len(all_frames_angles_z))]

    return np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)


def get_all_shapes_by_vid(open_face_path, video_path):
    os.makedirs(open_face_path, exist_ok=False)
    os.system('./OpenFace/build/bin/FeatureExtraction -3Dfp -pose -tracked -multi_view 1 -f "{0}" -out_dir "{1}"'.format
              (video_path, open_face_path))
    output_csv = '{0}/{1}.csv'.format(open_face_path, os.path.splitext(os.path.basename(video_path))[0])
    res = genfromtxt(output_csv, delimiter=',')

    xs = res[1:len(res), 11:79]
    ys = res[1:len(res), 79:147]
    zs = res[1:len(res), 147:215]
    shape = np.stack((xs, ys, zs), axis=-1)
    # shape = smooth_anchors(shape)

    head_angles = res[1:len(res), 8:11]
    head_angles = smooth_angles(head_angles)

    head_poses = res[1:len(res), 5:8]
    # head_poses = smooth_head_poses(head_poses)

    return shape, head_angles, head_poses


def calc_face_size(face_shape_3d, face_rotations, face_positions_3d, frame_data):
    total_face_size = 0
    # total_mouth_size = 0

    centered_kp_3d = face_shape_3d - face_positions_3d[:, np.newaxis, :]

    for i in range(0, len(face_shape_3d)):
        cur_head_pos_3d = face_positions_3d[i]
        cur_head_rot = face_rotations[i]
        cur_centered_3d_kp = centered_kp_3d[i]

        cur_r = euler_angles_to_rotation_matrix(cur_head_rot)

        cur_aligned_centered_kp3 = np.matmul(cur_centered_3d_kp, cur_r)
        cur_aligned_kp_3d = cur_aligned_centered_kp3 + cur_head_pos_3d[np.newaxis, :]

        cur_aligned_kp_2d = []

        for p3d in cur_aligned_kp_3d:
            if p3d[2] != 0:
                x = ((p3d[0] * frame_data.fx / p3d[2]) + frame_data.cx)
                y = ((p3d[1] * frame_data.fy / p3d[2]) + frame_data.cy)
            else:
                x = p3d[0]
                y = p3d[1]

            cur_aligned_kp_2d.append(np.array((x, y)))

        cur_aligned_kp_2d = np.array(cur_aligned_kp_2d)

        if cur_head_pos_3d[2] != 0:
            x = ((cur_head_pos_3d[0] * frame_data.fx / cur_head_pos_3d[2]) + frame_data.cx)
            y = ((cur_head_pos_3d[1] * frame_data.fy / cur_head_pos_3d[2]) + frame_data.cy)
        else:
            x = cur_head_pos_3d[0]
            y = cur_head_pos_3d[1]

        cur_head_pos_2d = np.array((x, y))

        cur_centered_aligned_kp_2d = cur_aligned_kp_2d - cur_head_pos_2d[np.newaxis, :]
        # cur_centered_aligned_kmouth_p_2d = cur_centered_aligned_kp_2d[48:]
        total_face_size += sum(np.linalg.norm(cur_centered_aligned_kp_2d, axis=1)) / len(cur_centered_aligned_kp_2d)
    # mouth_size = sum(np.linalg.norm(cur_centered_aligned_kmouth_p_2d, axis=1)) / len(cur_centered_aligned_kmouth_p_2d)
    # total_mouth_size += mouth_size

    total_face_size = total_face_size / len(face_shape_3d)
    # total_mouth_size = total_mouth_size / len(face_shape_3d)

    return total_face_size


def process_video_file_wrapper(args):
    return process_video_file(*args)


def process_video_file(index, item, total, config, db_config, gen_db_path, target_video_fps):
    video_start_time = time.time()
    folder_name, file_name = item.split('/')
    gen_folder_path = '{}{}/'.format(gen_db_path, folder_name)
    num_of_frames_path = '{0}{1}_{2}'.format(gen_folder_path, file_name, db_config.num_of_frames_file_name)
    base_open_face_path = '{0}{1}/'.format(gen_folder_path, db_config.open_face_folder)
    open_face_path = '{0}{1}'.format(base_open_face_path, file_name)
    video_path = '{0}con_{1}.mp4'.format(gen_folder_path, file_name)
    normalized_video_path = "{0}normalized_video_{1}.mp4".format(gen_folder_path, file_name)
    video_frames_file_path = '{}{}_{}.mp4'.format(gen_folder_path, file_name, db_config.video_frames_file_name)
    shapes_2d_file_path = '{}{}_{}'.format(gen_folder_path, file_name, db_config.shapes2d_file_name)
    wav_path = '{0}con_{1}.wav'.format(gen_folder_path, file_name)

    # if config.general.debug:
    #	video_temp_path = '{}{}/'.format(gen_folder_path, file_name)
    #	if not os.path.isdir(video_temp_path):
    #		os.makedirs(video_temp_path)

    if not os.path.isdir(gen_folder_path):
        raise Exception('Please run audio pre process first')

    if os.path.isdir(open_face_path):
        shutil.rmtree(open_face_path)

    vidcap = cv2.VideoCapture(video_path)
    all_shapes, angles, head_poses = get_all_shapes_by_vid(open_face_path, video_path)

    v_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_data = create_frame_data(v_width, v_height)
    face_size = calc_face_size(all_shapes, angles, head_poses, frame_data)
    target_face_size = db_config.target_face_size
    scale_factor = target_face_size / face_size
    normalization_width = int(scale_factor * v_width)
    normalization_height = int(scale_factor * v_height)

    if scale_factor < 1:
        pad_or_crop = 'pad={0}:{1}'.format(v_width, v_height)
    else:
        crop_top_left_x = (normalization_width - v_width) // 2
        crop_top_left_y = 0
        pad_or_crop = 'crop={0}:{1}:{2}:{3}'.format(v_width, v_height, crop_top_left_x, crop_top_left_y)

    error = os.system(
        'ffmpeg -loglevel error -hwaccel cuvid -i "{0}" -vf \"scale={1}:{2}, {3}\" "{4}"'.format(
            video_path,
            normalization_width,
            normalization_height,
            pad_or_crop,
            normalized_video_path))

    if error:
        raise Exception('error while normalizing face size')

    # assert_video_length(video_path, normalized_video_path)

    # print("face size successfully normalized")
    # print("detecting normalized face features")

    shutil.rmtree(open_face_path)
    vidcap = cv2.VideoCapture(normalized_video_path)

    all_shapes, angles, head_poses = get_all_shapes_by_vid(open_face_path, normalized_video_path)

    if not config.general.debug:
        os.remove(normalized_video_path)
        os.remove(video_path)
        # os.remove(wav_path)
        shutil.rmtree(open_face_path)

    frame_data = create_frame_data(int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    np.save(num_of_frames_path, number_of_frames)

    success = True
    counter = 0
    frames = []
    mouths = []
    shapes_2d = []
    error = False

    tmp_folder = '{}{}_tmp/'.format(gen_folder_path, file_name)
    os.makedirs(tmp_folder)

    while success:
        success, frame = vidcap.read()

        if not success:
            continue

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose = head_poses[counter]
        pose = np.expand_dims(pose, axis=0)

        current_frame_angles = angles[counter]
        # r = euler_angles_to_rotation_matrix(current_frame_angles)
        rz = euler_angles_to_rotation_matrix((0, 0, current_frame_angles[2]))

        face_3d = all_shapes[counter]
        face_3d_translated = face_3d - pose

        # face_3d_aligned_centered = np.matmul(face_3d_translated, r)
        face_3d_aligned_z_centered = np.matmul(face_3d_translated, rz)
        face_3d_aligned_z = face_3d_aligned_z_centered + pose

        face_2d_aligned_z = []

        for p3d in face_3d_aligned_z:
            if p3d[2] != 0:
                x = ((p3d[0] * frame_data.fx / p3d[2]) + frame_data.cx)
                y = ((p3d[1] * frame_data.fy / p3d[2]) + frame_data.cy)
            else:
                x = p3d[0]
                y = p3d[1]

            face_2d_aligned_z.append(np.array((x, y)))

        face_2d_aligned_z = np.array(face_2d_aligned_z).astype(np.int32)

        if pose[0, 2] != 0:
            x = ((pose[0, 0] * frame_data.fx / pose[0, 2]) + frame_data.cx)
            y = ((pose[0, 1] * frame_data.fy / pose[0, 2]) + frame_data.cy)
        else:
            x = pose[0, 0]
            y = pose[0, 1]

        pose_in_image = np.array((x, y))

        m = cv2.getRotationMatrix2D((pose_in_image[0], pose_in_image[1]),
                                    current_frame_angles[2] * 180.0 / np.pi, 1)
        frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_AREA)

        frames.append(frame)
        shapes_2d.append(face_2d_aligned_z)

        if config.general.image_gen_show_frames:
            # for (x, y) in face_2d_aligned_z:
            #   cv2.circle(frame, (int(np.rint(x)), (int(np.rint(y)))), 3, (255, 0, 0), -1)

            tmp_frame = frame.copy()

            cv2.circle(tmp_frame, (int(np.rint(face_2d_aligned_z[33][0])), int(np.rint(face_2d_aligned_z[33][1]))),
                       3,
                       (0, 0, 255), -1)
            cv2.circle(tmp_frame, (int(np.rint(face_2d_aligned_z[29][0])), int(np.rint(face_2d_aligned_z[29][1]))),
                       3,
                       (0, 255, 0), -1)
            cv2.circle(tmp_frame, (int(np.rint(pose_in_image[0])), int(np.rint(pose_in_image[1]))), 3,
                       (255, 0, 0), -1)

            # cv2.imshow("Frame", cv2.cvtColor(tmp_frame, cv2.COLOR_RGB2BGR))
            cv2.imshow("Frame", tmp_frame)
            cv2.waitKey(1) & 0xFF

        # validate mouth extraction (validate size)
        try:
            mouth = extract_mouth_from_frame(frame, face_2d_aligned_z[29], config.general.mouth_height, config.general.mouth_width)
            mouths.append(mouth)
        except:
            error = True
            # if os.path.exists(db_creation_errors_file_path):
            #     append_write = 'a'  # append if already exists
            # else:
            #     append_write = 'w'  # make a new file if not
            #
            # db_creation_errors_file = open(db_creation_errors_file_path, append_write)
            #
            # db_creation_errors_file.write(
            #     'extract_mouth_from_frame error in {}, frame:{}\n'.format(video_path, counter))
            #
            # db_creation_errors_file.close()
            break

        counter += 1
        cv2.imwrite('{}{}.png'.format(tmp_folder, counter), frame)

    # if config.general.debug:
    #	cv2.imwrite('{}frame_{}.png'.format(video_temp_path, counter), frame)
    #	cv2.imwrite('{}mouth_{}.png'.format(video_temp_path, counter), mouth)

    if config.general.image_gen_show_frames:
        print('in case of a parallel run, destroyAllWindows will probably kill other iterations windows')
        cv2.destroyAllWindows()

    if not error:
        error = os.system(
            'ffmpeg -loglevel error -hwaccel cuvid -r {} -f image2 -s {}x{} -i {}%d.png -vcodec libx264 -crf 15 -pix_fmt yuv420p {}'.format(
                target_video_fps,
                frames[0].shape[0],
                frames[0].shape[1],
                tmp_folder,
                video_frames_file_path
            ))

        if error:
            msg = 'error while compiling mouths images'
            print(msg)
            raise Exception(msg)

        assert_video_length_c(video_frames_file_path, number_of_frames)
        # np.save(video_frames_file_path, mouths)
        # np.save(video_frames_file_path, frames)
        np.save(shapes_2d_file_path, shapes_2d)

    shutil.rmtree(tmp_folder)

    total_video_time = time.time() - video_start_time
    print(
        '===== {}/{} video db for {} was created in {} seconds ======'.format(index + 1, total, item, total_video_time))


def create_video_db(config, db_config, gen_db_path, raw_db_list_path, target_video_fps):
    start_video_time = time.time()

    print('####### Creating video db #######')

    if not os.path.isdir(gen_db_path):
        raise Exception('Please run audio pre process first')

    video_fps_path = '{}{}'.format(gen_db_path, db_config.video_fps_file_name)
    np.save(video_fps_path, target_video_fps)

    with open(raw_db_list_path) as f:
        files_list = f.readlines()

    files_list = [x.strip() for x in files_list]

    num_cores = multiprocessing.cpu_count()

    with Pool(num_cores) as pool:
        params_p = [(index, item, len(files_list), config, db_config, gen_db_path, target_video_fps) for index, item in
                    enumerate(files_list)]
        #process_video_file(0, files_list[0], len(files_list), config, db_config, gen_db_path, target_video_fps)
        pool.map(process_video_file_wrapper, params_p)

    total_video_time = time.time() - start_video_time
    print('####### Video db was created in {} seconds #######'.format(total_video_time))


def main(argv):
    print('For TIMIT, you should run first preprocess_TIMIT_db.py, if you didnt run it, please stop current execution.')
    # time.sleep(15)
    (opts, args) = parser.parse_args(argv)
    config = ConfigParser(opts.config)
    db_config = eval('config.{}'.format(config.general.db_name))

    db_type = eval('DBType.{}'.format(config.general.db_gen_type))

    if db_type == DBType.Validation:
        gen_db_path = db_config.validation_path
        raw_db_path = db_config.raw_validation_path
        raw_db_list_path = db_config.raw_validation_list_path
    elif db_type == DBType.Test:
        gen_db_path = db_config.test_path
        raw_db_path = db_config.raw_test_path
        raw_db_list_path = db_config.raw_test_list_path
    elif db_type == DBType.Train:
        gen_db_path = db_config.train_path
        raw_db_path = db_config.raw_train_path
        raw_db_list_path = db_config.raw_train_list_path
    elif db_type == DBType.Example:
        gen_db_path = db_config.example_path
        raw_db_path = db_config.raw_example_path
        raw_db_list_path = db_config.raw_example_list_path

    # db_creation_errors_file_path = '{}{}'.format(gen_db_path, db_config.db_creation_errors_file_name)

    # if os.path.isfile(db_creation_errors_file_path):
    #    raise Exception('please delete {}'.format(db_creation_errors_file_path))

    create_audio_features(db_config, gen_db_path, raw_db_path, raw_db_list_path, config.general.target_audio_rate,
                          config.general.target_video_fps)

    create_video_db(config, db_config, gen_db_path, raw_db_list_path, config.general.target_video_fps)

    print('Please check db_creation_errors !!!!!')
    print('Please run Utils/generate_frames_from_videos.py')
    print('Please run Utils/test_db_shifts_file_generator.py for validation and test dbs (for single shift cases)')
    print('Please run Utils/generate_acnhors.py')
    print('Please run Model/generate_test_db.py')


if __name__ == '__main__':
    main(sys.argv)
