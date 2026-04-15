from smoothoperators import feature_profiles
from smoothoperators import event_filtering

if __name__ == '__main__':
    PATH_TO_IIOT_CLICKHOUSE = 'Your path to iiot clickhouse'
    event_filter = event_filtering.EventFilter(
        'features',
        'events',
        'join_key',
        'unix_seconds',
        'event_time_unix',
    )
    event_filter.thresholds = [
        1, 4, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2
    ]
    event_filter.half_lifes = ['minute']
    event_filter.num_trials = 1
    entity = 'dst_ip'

    kept_volume_day_ds = {
        f'thresholds_{t}': []
        for t in event_filter.thresholds
    }

    kept_volume_day_ds_vr = {
        f'thresholds_{t}': []
        for t in event_filter.thresholds
    }

    kept_volume_day_vanilla = {
        f'thresholds_{t}': []
        for t in event_filter.thresholds
    }
    with feature_profiles.ClickHouseConnector(PATH_TO_IIOT_CLICKHOUSE):
        for i in range(15):
            try:
                volumes = event_filter.kept_volume(
                    entity, i, double_stochastic=True).loc[0].to_dict()
                for t in event_filter.thresholds:
                    kept_volume_day_ds[f'thresholds_{t}'].append(volumes[
                        f'kept_volume_half_life_minute_threshold_{event_filtering.custom_sci(t)}']
                                                                 )
            except Exception as e:
                print(f'Failed to fetch trial {i} double stochastic')

        for i in range(15):
            try:
                volumes = event_filter.kept_volume(
                    entity, i, double_stochastic=True,
                    var_alpha=1).loc[0].to_dict()
                for t in event_filter.thresholds:
                    kept_volume_day_ds_vr[f'thresholds_{t}'].append(volumes[
                        f'kept_volume_half_life_minute_threshold_{event_filtering.custom_sci(t)}']
                                                                    )
            except Exception as e:
                print(
                    f'Failed to fetch trial {i} double stochastic with var reduction'
                )
                print(e)

        for i in range(15):
            try:
                volumes = event_filter.kept_volume(
                    entity, i, double_stochastic=False).loc[0].to_dict()
                for t in event_filter.thresholds:
                    kept_volume_day_vanilla[f'thresholds_{t}'].append(volumes[
                        f'kept_volume_half_life_minute_threshold_{event_filtering.custom_sci(t)}']
                                                                      )
            except Exception as e:
                print(f'Failed to fetch trial {i} vanilla')
                print(e)

    print('\nDOUBLE STOCHASTIC')
    print(kept_volume_day_ds)
    print('\nDOUBLE STOCHASTIC VARIANCE REDUCTION')
    print(kept_volume_day_ds_vr)
    print('\nVANILLA')
    print(kept_volume_day_vanilla)
