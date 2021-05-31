# betastar

Mastering StarCraft 2 minigames with Reinforcement Learning.

## Setup

1. Make sure you have Docker installed on your machine.
2. Install VSCode.
3. When you open the folder, it'll ask you to "Reopen in Container". Do it, and you're set!
4. You might need to re-run `pip install poetry` after starting the container.

## PySC2

Inside the container you can run different pysc2 commands, here are some useful commands:

* `python -m pysc2.bin.map_list` print the list of available maps
* `python -m pysc2.bin.valid_actions` print the list of valid actions
* `python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.random_agent.RandomAgent` play a game in the Simple64 map with the random agent (Python path to an agent class)

## Running agents

The easiest way to run an agent is within the Run & Debug panel in VSCode. In order to view the rendering of the agent you first need to connect to the VNC server at http://localhost:8080/vnc.html.

From the Run & Debug panel select the launch configuration you want to execute and click on the play button. You can also add or modify existing configurations by editing the `.vscode/launch.json` file.

You can also run an agent without debugging with `python -m betastar run` (use `run --help` to see the available options).

## Troubleshooting

### The VNC server seems to have shut down. How do I restart the whole graphics thing?

Although rebuilding the container will work, issuing this command in a VSCode terminal (which runs inside the container) is easier:

```bash
supervisord -c /workspaces/betastar/.devcontainer/supervisord.conf
```