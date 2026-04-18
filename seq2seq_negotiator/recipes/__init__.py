
from .single_target import SingleTargetRecipe
from .multitask import MultiTaskRecipe

RECIPE_REGISTRY = {
    "single_target": SingleTargetRecipe(),
    "multitask": MultiTaskRecipe(),
}
