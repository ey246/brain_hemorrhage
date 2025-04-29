import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from fractions import Fraction
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

"""
Preprocessing the images and loading the datasets
"""
def preprocess(image_path, mask_path):
    # read gray scale CT image and resize and turn it into RGB for VGG16
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    # read the mask file and turn it into a binary representation
    # (1 for hemorrhage and 0 for not)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 127).astype(np.float32)

    return img_rgb, mask[..., np.newaxis]

def load_dataset(image_dir, mask_dir):

    # loads a dataset in sorted order to make finding patients easier
    # since patient CTs are stored as ID_(slice #)
    image_files = sorted(os.listdir(image_dir))
    X, Y = [], []

    for fname in image_files:
        image_path = os.path.join(image_dir, fname)
        mask_name = fname.replace('.jpg', '_HGE_Seg.jpg')
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            img, mask = preprocess(image_path, mask_path)
            X.append(img)
            Y.append(mask)

    return np.array(X), np.array(Y)

IMG_SIZE = 224 # resize all of our images to 224 to work with vgg16
NUM_ACTIONS = 9 
HISTORY_SIZE = 0 # customizable to situation, currently set at 0
STATE_DIM = 4096 + NUM_ACTIONS * HISTORY_SIZE
MAX_STEPS = 30
BATCH_SIZE = 300 # size of batch for experience replay
MAX_EXPERIENCE_SIZE = 10000 # optional if you want to cap replay buffer size
GAMMA = 0.99 # discount factor

"""
This is a section on processing auxiliaries which include a functino for extracting features,
cropping images, computing IoU, and how the agent moves.
"""

@tf.function(reduce_retracing=True)
def extract_feature_batch_tf(images, histories, feature_extractor):

    # reshape images to a size VGG16 can work with
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
    images = tf.ensure_shape(images, [None, IMG_SIZE, IMG_SIZE, 3])  # batch dimension = None

    # one-hot encode the histories which should have dimensions (B, HISTORY_SIZE, NUM_ACTIONS)
    histories_onehot = tf.one_hot(histories, depth=NUM_ACTIONS, dtype=tf.float32)
    histories_flat = tf.reshape(histories_onehot, [tf.shape(histories)[0], -1])

    # extract features
    features = feature_extractor(images)
    features_flat = tf.reshape(features, [tf.shape(images)[0], -1])

    return tf.concat([features_flat, histories_flat], axis=1) 

def compute_mask(action, box):

    # move the box by some multiplicative alpha of the entire length of box
    # (i.e. alpha*length of current box prediction)
    alpha = 0.2
    delta_w = alpha * (box[2] - box[0])
    delta_h = alpha * (box[3] - box[1])
    x1, y1, x2, y2 = box

    if action == 0:  # move right
        x1 += delta_w
        x2 += delta_w
    elif action == 1:  # move left
        x1 -= delta_w
        x2 -= delta_w
    elif action == 2:  # move up
        y1 -= delta_h
        y2 -= delta_h
    elif action == 3:  # move down
        y1 += delta_h
        y2 += delta_h
    elif action == 4:  # zoom in (keeping proportions the same)
        x1 += delta_w
        x2 -= delta_w
        y1 += delta_h
        y2 -= delta_h
    elif action == 5:  # zoom out (keeping proportions the same)
        x1 -= delta_w
        x2 += delta_w
        y1 -= delta_h
        y2 += delta_h
    elif action == 6:  # vertical zoom (squish height)
        y1 += delta_h
        y2 -= delta_h
    elif action == 7:  # horizontal zoom (squish width)
        x1 += delta_w
        x2 -= delta_w

    # clip box to ensure everything is valid
    x1, x2 = np.clip([x1, x2], 0, IMG_SIZE)
    y1, y2 = np.clip([y1, y2], 0, IMG_SIZE)

    # make sure we dont invert the box by accident
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    
    # determine coordinates of intersection box
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # calculate area of intersection box
    inter = max(0, xB - xA) * max(0, yB - yA)

    # compute union area
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # return intersection divided by union
    return inter / float(areaA + areaB - inter + 1e-6)

def crop_image(img, box):
    x1, y1, x2, y2 = map(int, box)

    #crop image by just taking a slice of y and x values
    cropped = img[y1:y2+1, x1:x2+1]
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE)) if cropped.size else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


"""
This is a section on rewards. Currently we define 3 different functions:

binary - give reward based on whether or not IoU gets better (+1 if better, and -1 if worse), along with specific time out and quitting rewards
IoU difference - give a reward based on new IoU being better or worse than previous IoU (motivated by wanting binary reward to converge faster)
IoU penalty area - gives a reward based purely on new IoU but adds a penalty factor that is proportional to how small the bounding box is (i.e. give it less penalty for predicting smaller boxes), motivated by the agent sometimes just zooming out
Manhattan Distance - gives a reward based on how much the Manhattan distance has improved (thought this might be good to separate moving in x direction from moving in the y direction), penalizing moving too inward
coordinates - give a fixed reward for moving box closer to the ground truth box and penalizes for shifting inwards (motivated by binary and IoU diff producing lines and Manhattan distance giving rewards with too high variance)

Each of the these take in as inputs: action, gt (ground truth), box (current prediction), and timed_out (to siginal if timed out).
Remember that each box is represented as two coordinates, x1 and y1 for the top left corner and x2 and y2 for the bottom right corner.
"""

def reward_binary(action, gt, box, timed_out):
    # compute current and new iou scores to compare to
    new_iou = compute_iou(compute_mask(action, box), gt)
    curr_iou = compute_iou(box, gt)

    # if agent quits early, give rewards
    if action == 8:
        if new_iou >= 0.2:
            reward = 5
        else:
            reward = -5

    # else if timed out, give small penalty
    elif timed_out :
        reward = -2
    
    # else give +1 or -1 based on improved iou score
    else: 
        if new_iou > curr_iou:
            reward = 1
        else:
            reward = -1

    return reward

def reward_iou_diff(action, gt, box, timed_out):
    # compute iou scores
    new_iou = compute_iou(compute_mask(action, box), gt)
    curr_iou = compute_iou(box, gt)

    # if agent terminates itself give rewards
    if action == 8:
        if new_iou >= 0.2:
            reward = 5
        else:
            reward = -3

    # else if agent times out, no extra rewards are given
    elif timed_out:
        reward = 0

    # else get reward based on how much iou has improved (or not)
    else:
        reward = new_iou - curr_iou

    return reward

def reward_iou_area_penalty(action, gt, box, timed_out):

    # compute new iou and mask
    new_box = compute_mask(action, box)
    new_iou = compute_iou(new_box, gt)

    # calculate area of new mask then normalize it to image size
    area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
    norm_area_penalty = area / (IMG_SIZE * IMG_SIZE)

    # typical penalization for timing out versus quitting early
    if timed_out:
        reward = 0
    elif action == 8:
        if new_iou > 0.2:
            reward = 5
        else:
            reward = -3
    else:
        # if agent doesn't quit or time out, compute reward based on how good new IoU score is
        # minus the penalty where the pentalty is smaller based on how small the new mask is
        reward = new_iou - 0.1 * norm_area_penalty
    return reward

def reward_manhattan_distance(action, gt, box, end=False, timed_out=False):

    # compute new mask and extract all coordinates from gt, old, and new mask
    new_box = compute_mask(action, box)
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    old_x1, old_y1, old_x2, old_y2 = box
    new_x1, new_y1, new_x2, new_y2 = new_box

    # Manhattan distance to each corner
    old_dist = abs(old_x1 - gt_x1) + abs(old_y1 - gt_y1) + abs(old_x2 - gt_x2) + abs(old_y2 - gt_y2)
    new_dist = abs(new_x1 - gt_x1) + abs(new_y1 - gt_y1) + abs(new_x2 - gt_x2) + abs(new_y2 - gt_y2)

    improvement = old_dist - new_dist

    # Penalize if predicted box goes inward past the ground truth
    inward_penalty = 0
    if new_x1 > gt_x1:
        inward_penalty += new_x1 - gt_x1
    if new_y1 > gt_y1:
        inward_penalty += new_y1 - gt_y1
    if new_x2 < gt_x2:
        inward_penalty += gt_x2 - new_x2
    if new_y2 < gt_y2:
        inward_penalty += gt_y2 - new_y2

    reward = improvement - 2 * inward_penalty

    if end:
        if timed_out:
            reward -= 5  # timeout penalty
        else:
            iou = compute_iou(new_box, gt)
            if iou > 0.6:
                reward += 1000  # high bonus for good box (has to be high due to distances giving big rewards)
            elif iou > 0.25:
                reward += 500   # moderate bonus
            else:
                reward -= 1000   # penalty if IoU still poor

    return reward

def reward_coordinates(action, gt, box, timed_out):

    # get coordinates for the new box, current box, and ground truth box
    new_box = compute_mask(action, box)
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    curr_x1, curr_y1, curr_x2, curr_y2 = box
    new_x1, new_y1, new_x2, new_y2 = new_box

    # if agent quits, give it rewards for being good, ok, and bad
    # if action == 8:
    #     iou = compute_iou(new_box, gt)
    #     if iou >= 0.2:
    #         reward = 25
    #     elif iou >= 0.05:
    #         reward = 15
    #     else:
    #         reward = -20
    #     return reward

    # trying something new to see if increasing reward linearly will help it improve
    # instead of harsh steps in the reward function
    if action == 8:
        iou = compute_iou(new_box, gt)
        if iou < 0.05:
            reward = -20
        elif iou <= 0.4:
            reward = 10 + (iou - 0.05) * (15 / (0.4 - 0.05))
        else:
            reward = 45
        return reward

    # else give it a reward based on whether or not each coordinate for the box has improved
    reward = 0

    curr_coords = [curr_x1, curr_y1, curr_x2, curr_y2]
    new_coords = [new_x1, new_y1, new_x2, new_y2]
    gt_coords = [gt_x1, gt_y1, gt_x2, gt_y2]

    # assign True to the top left corners and False to the bottom right corners
    for old_c, new_c, gt_c, is_min_corner in zip(curr_coords, new_coords, gt_coords, [True, True, False, False]):

        # if agent moves closer, it gets a +1 reward else -1
        if abs(new_c - gt_c) < abs(old_c - gt_c):
            reward += 1
        else:
            reward -= 1
        
        # if agent minimizes beyond the gt box, it incurs extra losses
        if (is_min_corner and new_c > gt_c) or (not is_min_corner and new_c < gt_c):
            reward -= 2
        
        # if agent predicts completely correctly a coordinate (c), give it an extra reward
        if new_c == gt_c:
            reward += 2

    # penalize for timing out
    if timed_out:
        reward -= 5

    return reward

"""
This is the section concerning the deep Q network
"""
def create_q_model():
    # define a neural network that takes in feature map (extracted by vgg16) and outputs Q values for each action
    model = Sequential([
        Input(shape=(STATE_DIM,)),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(NUM_ACTIONS)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

class QAgent:
    def __init__(self, model, num_actions = NUM_ACTIONS, gamma = GAMMA, epsilon = 1.0, max_memory_size = MAX_EXPERIENCE_SIZE):
        self.model = model
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = []
        self.max_memory_size = max_memory_size

    def remember(self, transition):
        # add experience to the replay buffer
        self.memory.append(transition)
        # if over flowing, discard oldest experience
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]

    def select_action(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        # If 1D, add batch dimension: (1, N)
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size = BATCH_SIZE):
        if len(self.memory) == 0:
            return

        batch_size = min(batch_size, len(self.memory))
        batch = random.sample(self.memory, batch_size)

        # Unpack batch
        states, actions, rewards, next_states, end_bool = zip(*batch)

        # Stack into batched tensors
        states = tf.stack(states)
        next_states = tf.stack(next_states)

        # Predict Q-values for current states and next states
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        target_qs = q_values.copy()

        for i in range(batch_size):
            target = rewards[i]
            if not end_bool[i]:
                target += self.gamma * np.max(next_q_values[i])
            target_qs[i][actions[i]] = target

        return self.model.train_on_batch(states, target_qs, return_dict=True)

"""
The training function
"""

import matplotlib.pyplot as plt
import contextlib

def train_dqn(X_train, Y_train, vgg16_model, reward_fn, log_file, img_file, epochs, warmup_x, warmup_step, epsilon=1.0, batch_size=BATCH_SIZE):
    feature_extractor = tf.keras.Model(
        inputs=vgg16_model.input,
        outputs=vgg16_model.layers[20].output
    )
    feature_extractor.trainable = False  # freeze the feature extractor

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_q_model()
        agent = QAgent(model, num_actions=NUM_ACTIONS, epsilon=epsilon)

    loss_history = []
    reward_history = []
    total_step = 0

    with open(log_file, "w") as f, contextlib.redirect_stdout(f):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            for i in range(len(X_train)):
                img, mask = X_train[i], Y_train[i]
                ys, xs = np.where(mask.squeeze() > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                gt_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

                original_img = img

                history = [-1] * HISTORY_SIZE
                current_box = [0, 0, IMG_SIZE, IMG_SIZE]

                img_tensor = tf.convert_to_tensor(original_img, dtype=tf.float32)
                img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
                img_tensor = tf.expand_dims(img_tensor, axis=0)

                history_tensor = tf.constant([history], dtype=tf.int32)
                state = extract_feature_batch_tf(img_tensor, history_tensor, feature_extractor)[0]

                timed_out, step = False, 0
                done = False
                episode_reward = 0

                while not timed_out:
                    if total_step < warmup_step:
                        action = np.random.randint(NUM_ACTIONS)  # make it random early on when we're just collecting random steps
                    else:
                        action = agent.select_action(state)
                    if action == 8:
                        done = True
                        next_box = current_box
                    else:
                        next_box = compute_mask(action, current_box)

                    timed_out = step >= MAX_STEPS
                    reward = reward_fn(action, gt_box, current_box, timed_out=timed_out)

                    episode_reward += reward

                    print(f'Image {i}, Step {step}, Action {action}, Reward {reward:.4f}')
                    f.flush()

                    if HISTORY_SIZE > 0:
                        history = history[1:] + [action]
                        history_tensor = tf.constant([history], dtype=tf.int32)
                    else:
                        history_tensor = tf.constant([], shape=(1, 0), dtype=tf.int32)

                    next_crop = crop_image(original_img, next_box)
                    next_crop_tensor = tf.convert_to_tensor(next_crop, dtype=tf.float32)
                    next_crop_tensor = tf.image.convert_image_dtype(next_crop_tensor, tf.float32)
                    next_crop_tensor = tf.expand_dims(next_crop_tensor, axis=0)

                    next_state = extract_feature_batch_tf(next_crop_tensor, history_tensor, feature_extractor)[0]

                    agent.remember((state, action, reward, next_state, timed_out))

                    # train if total steps are less than the warm-up steps
                    if total_step > warmup_step:
                        train_result = agent.train(batch_size=batch_size)
                        if train_result:
                            loss_history.append(train_result['loss'])

                    if done:
                        break
                    state = next_state
                    current_box = next_box
                    step += 1
                    total_step += 1

                reward_history.append(episode_reward) 

            # decay epsilon exponentially after each epoch
            if epoch + 1 > warmup_x:
                agent.epsilon = max(agent.epsilon * 0.9, 0.05)

    # === Plotting Section ===
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Loss Plot
    axs[0].plot(loss_history)
    axs[0].set_xlabel("Training Step")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("DQN Training Loss Over Time")
    axs[0].grid(True)

    # Reward Plot
    axs[1].plot(reward_history)
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Total Reward")
    axs[1].set_title("Total Reward per Episode Over Time")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(img_file)
    plt.show()

    return agent.model


if __name__ == "__main__":
    train_img_dir = 'pre-split/train/images'
    train_mask_dir = 'pre-split/train/masks'
    test_img_dir = 'pre-split/test/images'
    test_mask_dir = 'pre-split/test/masks'

    # train_img_dir = 'mini_dataset/train/images'
    # train_mask_dir = 'mini_dataset/train/masks'
    # test_img_dir = 'mini_dataset/test/images'
    # test_mask_dir = 'mini_dataset/test/masks'

    X_train, Y_train = load_dataset(train_img_dir, train_mask_dir)
    X_test, Y_test = load_dataset(test_img_dir, test_mask_dir)

    vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    feature_extractor = tf.keras.Model(inputs=vgg16.input, outputs=vgg16.layers[20].output)

    model = train_dqn(X_train, Y_train, feature_extractor, reward_fn = reward_coordinates, log_file = 'training_log_coord_full.txt', img_file = 'plots_coord_full.png', epochs= 35, warmup_x = 0, warmup_step=300)
    model.save('dqn_full_model_coord.keras')