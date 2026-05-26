import pyglet

# === ENUMERATE DISPLAYS ===
# In pyglet 2.x, use pyglet.display.get_display() instead of pyglet.canvas
display = pyglet.display.get_display()
screens = display.get_screens()
for i, screen in enumerate(screens):
    print(f"Screen {i}: {screen.width}x{screen.height} at ({screen.x}, {screen.y})")

# === PICK THE PROJECTOR SCREEN ===
# Typically index 1 is the external display (the projector).
# If you have only one display, use 0. Adjust based on the print output above.
PROJECTOR_SCREEN_INDEX = 1

# === BLOCK SIZE ===
BLOCK_SIZE = 1000  # 100x100 pixel red square in the center

# === SETUP WINDOW ===
projector_screen = screens[PROJECTOR_SCREEN_INDEX]
window = pyglet.window.Window(
    fullscreen=True,
    screen=projector_screen
)

# Compute the center
center_x = window.width // 2
center_y = window.height // 2

# Create a red square
batch = pyglet.graphics.Batch()
red_square = pyglet.shapes.Rectangle(
    x=center_x - BLOCK_SIZE // 2,
    y=center_y - BLOCK_SIZE // 2,
    width=BLOCK_SIZE,
    height=BLOCK_SIZE,
    color=(255, 0, 0),
    batch=batch
)

@window.event
def on_draw():
    window.clear()
    batch.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE or symbol == pyglet.window.key.Q:
        pyglet.app.exit()

pyglet.app.run()