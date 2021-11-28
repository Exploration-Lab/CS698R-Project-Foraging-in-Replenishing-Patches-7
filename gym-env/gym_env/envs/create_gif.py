import imageio
import os
import re

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

filenames = sorted(os.listdir(os.path.join(__location__, "frame_save/random/",)))
filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
images = []
for filename in filenames:
    images.append(
        imageio.imread(os.path.join(__location__, "frame_save/random/" + filename))
    )
imageio.mimsave(
    os.path.join(__location__, "frame_save/random_agent_fps30.gif"), images, fps=30,
)


# filenames = sorted(os.listdir(os.path.join(__location__, "frame_save/max/",)))
# filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
# images = []
# for filename in filenames:
#     images.append(
#         imageio.imread(os.path.join(__location__, "frame_save/max/" + filename))
#     )
# imageio.mimsave(
#     os.path.join(__location__, "frame_save/max_agent_fps30.gif"), images, fps=30,
# )

# filenames = sorted(os.listdir(os.path.join(__location__, "frame_save/memory/",)))
# filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
# images = []
# for filename in filenames:
#     images.append(
#         imageio.imread(os.path.join(__location__, "frame_save/memory/" + filename))
#     )


# imageio.mimsave(
#     os.path.join(__location__, "frame_save/memory_agent_fps30.gif"), images, fps=30,
# )

# filenames = sorted(os.listdir(os.path.join(__location__, "frame_save/temporal/",)))
# filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
# images = []
# for filename in filenames:
#     images.append(
#         imageio.imread(os.path.join(__location__, "frame_save/temporal/" + filename))
#     )
# imageio.mimsave(
#     os.path.join(__location__, "frame_save/temporal_agent_fps30.gif"), images, fps=30,
# )
