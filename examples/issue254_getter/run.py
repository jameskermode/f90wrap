import itest

options = itest.kimdispersionequation_module.OptionsType()
options.omega = 0.5

obj = itest.kimdispersion_horton_module.KIMDispersion_Horton()
obj.initialize(options)

assert(abs(obj.options.omega - 0.5) < 1e-13)
