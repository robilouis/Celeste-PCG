import seleCte.celeskeleton.celeskeleton as celeskeleton

lvl_skel = celeskeleton.load_data_to_celeskeleton(
    "../pcg/pcg_model_results/default_generated_level"
)
lvl_skel.is_playable()
