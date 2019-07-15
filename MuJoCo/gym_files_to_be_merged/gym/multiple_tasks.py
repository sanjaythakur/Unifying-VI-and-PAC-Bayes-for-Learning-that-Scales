import gym, numpy as np, argparse


def get_HalfCheetah_task_by_mass(task_identity):
    env = gym.make('HalfCheetah-v1')
    body_mass = env.env.model.body_mass
    body_mass = np.array(body_mass)

    if task_identity == '0':
        # Keeping the default settings
        pass
    elif task_identity == '1':
        body_mass[2,0] = 0.7
        body_mass[3,0] = 0.4
    else:
        print('HalfCheetah-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    env.env.model.body_mass = body_mass
    return env


def get_Swimmer_task_by_length(task_identity):
    if task_identity == '0':
        # Keeping the default settings
        env = gym.make('Swimmer-v1')
    elif task_identity == '1':
        env = gym.make('Swimmer_1-v1')
    elif task_identity == '2':
        env = gym.make('Swimmer_2-v1')
    elif task_identity == '3':
        env = gym.make('Swimmer_3-v1')
    elif task_identity == '4':
        env = gym.make('Swimmer_4-v1')
    elif task_identity == '5':
        env = gym.make('Swimmer_5-v1')
    elif task_identity == '6':
        env = gym.make('Swimmer_6-v1')
    elif task_identity == '7':
        env = gym.make('Swimmer_7-v1')
    elif task_identity == '8':
        env = gym.make('Swimmer_8-v1')
    elif task_identity == '9':
        env = gym.make('Swimmer_9-v1')
    else:
        print('Swimmer-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)

    return env



def get_Swimmer_task_by_mass_and_length(task_identity):
    if task_identity == '0':
        # Keeping the default settings
        env = gym.make('Swimmer-v1')
    elif task_identity == '1':
        env = gym.make('Swimmer_3-v1')
        body_mass = env.env.model.body_mass
        body_mass = np.array(body_mass)
        body_mass[2,0] = 28.
    else:
        print('Swimmer-v1 not set for task_identity ' + task_identity + '. Program is exiting now...')
        exit(0)
    return env


def get_task_on_MUJOCO_environment(env_name, task_identity):
    if env_name == 'Swimmer':
        env = get_Swimmer_task_by_mass_and_length(task_identity=task_identity)
    elif env_name == 'HalfCheetah':
        env = get_HalfCheetah_task_by_mass(task_identity=task_identity)
    else:
        print('Multiple tasks not set for ' + env_name + '. Program is exiting now...')
        exit(0)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Get modified MUJOCO environment based on your task_identity specified'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-ti', '--task_identity', type=str,
                        help='The underlying and typically unobservable task_identity during operation of a controller',
                        default='0')
    args = parser.parse_args()
    
    env = get_task_identityual_MUJOCO_environment(**vars(args))
    print(env.env.model.body_mass)