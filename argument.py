# -*- coding: utf-8 -*-



def add_arguments(parser):
    """
    Add your arguments here.
    """
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_memory_size', type=int, default=400000)
    parser.add_argument('--update_target', type=int, default=5000)
    parser.add_argument('--update_current', type=int, default=4)
    parser.add_argument('--load_saver', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='save_dqn/')

    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--gamma_reward_decay', type=float, default=0.99)
    parser.add_argument('--observe_steps', type=int, default=5000)
    parser.add_argument('--explore_steps', type=int, default=1000000)
    parser.add_argument('--max_num_steps', type=int, default=10000)
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--saver_steps', type=int, default=25000)
    parser.add_argument('--num_eval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--output_logs', type=str, default='loss.csv')
    parser.add_argument('--resume', type=str, default=None)

    return parser
