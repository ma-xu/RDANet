from flops_counter import get_model_complexity_info
import models as models
customized_models_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print('-'.ljust(40,'-'))
for modelname in customized_models_names:
    if '50' in modelname:
        model = models.__dict__[modelname]()
        flops, params = get_model_complexity_info(model, (224, 224), as_strings=True, print_per_layer_stat=False)
        fixname=modelname.ljust(18,' ')
        info = fixname+'\t'+params+'\t'+flops
        print(info)
        print('-'.ljust(40,'-'))
