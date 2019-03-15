from models_integration_library.public_interface import start_loop

from inference import initialize_model, infer

initialize_model()

start_loop(infer)
