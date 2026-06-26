import argparse
import os, json, sys
import numpy as np
import torch
import utils
from Game import Game


def _setup_reflection(args, game) -> None:
    """
    Attach the reflection system to the game if --reflection is set.
    Silently skips when the memory module is unavailable.
    """
    if not getattr(args, "reflection", False):
        return
    try:
        from memory.reflection import QwenReflector
        from memory.memory_buffer import ReflectionMemory
    except ImportError:
        print("[main] WARNING: memory module not found — reflection disabled.")
        return

    reflector = QwenReflector(
        backend=args.reflection_backend,
        model=args.reflection_model or None,
        api_key=getattr(args, "reflection_api_key", None),
        temperature=args.reflection_temperature,
    )
    memory = ReflectionMemory(
        maxlen=args.reflection_maxlen,
        success_only=args.reflection_success_only,
        top_k=args.reflection_top_k,
    )

    # Load saved memory if requested
    if getattr(args, "reflection_load", None):
        memory.load(args.reflection_load)

    game.set_reflection_system(reflector, memory)

    # Also set up the planner LLM for online planning if requested
    if getattr(args, "llm_backend", None) and args.llm_backend != "offline":
        try:
            from utils.qwen_llm import QwenLLM
            qwen = QwenLLM(
                backend=args.llm_backend,
                model=args.llm_model or None,
                api_key=getattr(args, "llm_api_key", None),
            )
            game.teacher_policy.planner.set_llm(qwen)
            print(f"[main] Planner LLM set to {args.llm_backend}/{args.llm_model or 'default'}")
        except Exception as exc:
            print(f"[main] Could not set planner LLM: {exc}")


def train(args):
    for i in args.seed_list:
        args.savedir = args.savedir + "-" + str(i)
        args.seed = i
        game = Game(args)
        _setup_reflection(args, game)
        game.train()
        
def evaluate(args):
    assert args.loaddir
    print("env name: %s for %s" %(args.task, args.loaddir))
    args.seed = args.seed_list[0]
    game = Game(args, training=False)
    eval_returns = []
    eval_lens = []
    eval_success = []

    if len(args.env_seed_list) == 0:
        env_seed_list = [None] * args.num_eval
    elif len(args.env_seed_list) == 1:
        env_seed_list = [args.env_seed_list[0] + i for i in range(args.num_eval)]
    else:
        env_seed_list = args.env_seed_list

    for i in env_seed_list:
        eval_outputs = game.evaluate(seed = i, teacher_policy = args.eval_teacher)
        eval_returns.append(eval_outputs[0])
        eval_lens.append(eval_outputs[1])
        eval_success.append(eval_outputs[2])

    print("Mean return:", np.mean(eval_returns))
    print("Mean length:", np.mean(eval_lens))
    print("Success rate:", np.mean(eval_success))


def render(args):
    """
    Render episodes in a live cv2 window with action + plan overlay.

    Example
    -------
        python main.py render --task SimpleDoorKey --loaddir run2 `
            --loadmodel acmodel --n_render 3 --fps 5
    """
    from simulator.render_episode import run_render

    # A loaded model is optional — we can render a random/teacher policy too
    args.seed = args.seed_list[0]
    game = Game(args, training=False)

    # Optional: attach reflection memory (read-only) so planner gets context
    _setup_reflection(args, game)

    # Build save directory next to the model log (or a tmp dir)
    if args.loaddir:
        save_dir = os.path.join(
            args.logdir, args.policy, args.task, args.loaddir, "render_output"
        )
    else:
        save_dir = os.path.join("render_output", args.task)

    print(f"\n[render] Task        : {args.task}")
    print(f"[render] Policy      : {'teacher' if args.render_teacher else 'student'}")
    print(f"[render] Episodes    : {args.n_render}")
    print(f"[render] FPS         : {args.render_fps}")
    print(f"[render] Show window : {not args.no_window}")
    print(f"[render] Save dir    : {save_dir}\n")

    run_render(
        game,
        n_episodes=args.n_render,
        fps=args.render_fps,
        show_window=not args.no_window,
        save_dir=save_dir,
        teacher_policy=args.render_teacher,
    )


if __name__ == "__main__":
    utils.print_logo(subtitle="Maintained by Research Center for Applied Mathematics and Machine Intelligence, Zhejiang Lab")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SimpleDoorKey", help="SimpleDoorKey, KeyInBox, RandomBoxKey, ColoredDoorKey, DynamicDoorKey") 
    
    # parser.add_argument("--env_seed", type=int, default=0)
    parser.add_argument("--env_seed_list", type=int, nargs="*", default=[0], help="Seeds for evaluation environments")
    parser.add_argument("--seed_list", type=int, nargs="*", default=[0], help="Seeds for Numpy, Torch and LLM")
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy", type=str, default='ppo')
    parser.add_argument("--n_itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--traj_per_itr", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--recurrent", default=False, action='store_true')
    
    parser.add_argument("--logdir", type=str, default="log") # Where to log diagnostics to
    parser.add_argument("--loaddir", type=str, default=None)
    parser.add_argument("--loadmodel", type=str, default="acmodel")
    parser.add_argument("--savedir", type=str, required=True, help="path to folder containing policy and run details")
    
    parser.add_argument("--offline_planner", default=False, action='store_true')
    parser.add_argument("--soft_planner", default=False, action='store_true')
    parser.add_argument("--eval_teacher", default=False, action='store_true')
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)

    # ── Reflection system ──────────────────────────────────────────────────────
    parser.add_argument("--reflection", default=False, action='store_true',
                        help="Enable reflection-memory system")
    parser.add_argument("--reflection_backend", type=str, default="offline",
                        choices=["offline", "ollama", "dashscope"],
                        help="Backend for the reflector LLM")
    parser.add_argument("--reflection_model", type=str, default=None,
                        help="Reflector model name (e.g. 'qwen2.5:7b')")
    parser.add_argument("--reflection_api_key", type=str, default=None,
                        help="API key for dashscope reflector backend")
    parser.add_argument("--reflection_temperature", type=float, default=0.4,
                        help="Temperature for reflector LLM (default: 0.4)")
    parser.add_argument("--reflection_maxlen", type=int, default=20,
                        help="Max number of reflections to keep in memory")
    parser.add_argument("--reflection_top_k", type=int, default=5,
                        help="How many recent reflections to inject into planner")
    parser.add_argument("--reflection_success_only", default=False, action='store_true',
                        help="Only store reflections from successful episodes")
    parser.add_argument("--reflection_load", type=str, default=None,
                        help="Path to a saved reflection_memory.json to load on startup")

    # ── Planner LLM (online mode) ──────────────────────────────────────────────
    parser.add_argument("--llm_backend", type=str, default=None,
                        choices=["offline", "ollama", "dashscope"],
                        help="Backend for the planner LLM (overrides offline_planner)")
    parser.add_argument("--llm_model", type=str, default=None,
                        help="Planner model name (e.g. 'qwen2.5:3b')")
    parser.add_argument("--llm_api_key", type=str, default=None,
                        help="API key for dashscope planner backend")
    
    # ── Render flags (only registered for 'render' sub-command) ──────────────
    if len(sys.argv) > 1 and sys.argv[1] == 'render':
        parser.add_argument("--n_render",       type=int,   default=3,
                            help="Number of episodes to render (default: 3)")
        parser.add_argument("--render_fps",     type=int,   default=5,
                            help="Frames per second for live window and saved video (default: 5)")
        parser.add_argument("--render_teacher", default=False, action='store_true',
                            help="Use teacher policy instead of trained student policy")
        parser.add_argument("--no_window",      default=False, action='store_true',
                            help="Disable live cv2 window (save video only)")

    if sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        evaluate(args)
    elif sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        train(args)
    elif sys.argv[1] == 'render':
        sys.argv.remove(sys.argv[1])
        args = parser.parse_args()
        render(args)
    else:
        print("Invalid option '{}'".format(sys.argv[1]))