import importlib

def find_trainer_using_name(model_name):
    model_filename = "trainers." + model_name + "_trainer"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'trainer'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def create_trainer(opt):
    model = find_trainer_using_name(opt.trainer)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))

    return instance
