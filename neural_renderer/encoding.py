def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, 
                log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'hashgrid_diff':
        from hashencoder.hashgrid import HashEncoder
        encoder = HashEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, 
                    per_level_scale=2, base_resolution=base_resolution, 
                    log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution)

    elif encoding == 'frequency':
        if multires == 0:
            return lambda x, **kwargs: x, input_dim
        #encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'integrated_dir':
        from ide_encoder import IntegratedDirEncoder
        encoder = IntegratedDirEncoder(input_dim=input_dim, deg_view=degree)

    else:
        raise NotImplementedError('Unknown encoding mode')

    return encoder, encoder.output_dim
