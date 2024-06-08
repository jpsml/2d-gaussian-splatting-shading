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

    else:
        raise NotImplementedError('Unknown encoding mode')

    return encoder, encoder.output_dim
