from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm


def plot_arrival_intensity(df: pd.DataFrame,
                           entity: str,
                           half_life: Union[str, float],
                           included: str,
                           output_path: str,
                           n: int = 1):
    """Plots the estimations for the intenesity of arriving events for
    a given entity

    Args:
      df: Pandas dataframe. Must have an `event_time_unix` column with
        timestamps in miliseconds, an `entity` columns, an `included column`
        and an `intensity_half_life_{half_life}` column.
      entity: Entity for which to plot, e.g, merchant_id, card_id, device_ip...
      half_life: The half_life for which to plot the intensity function.
      included: Column name that decides if an event was filtered out or not.
      output_path: Path to where the plot should be saved.
      n: Rank of the entity. n=1 we choose the entity with highest volume.
    """
    df['timestamp'] = pd.to_datetime(df['event_time_unix'], unit='ms')
    entity_id = df[entity].value_counts().nlargest(n).index.tolist()[-1]
    entity_df = df[df[entity].isin([entity_id])]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    intensity_col = f'intensity_half_life_{half_life}'
    if included and included in entity_df.columns:
        filtered = entity_df[entity_df[included] == 0]
        kept = entity_df[entity_df[included] == 1]

        ax1.scatter(filtered['timestamp'],
                    filtered[intensity_col],
                    color='red',
                    marker='.',
                    s=1,
                    label=f'{entity} {entity_id}; excluded')

        ax1.scatter(kept['timestamp'],
                    kept[intensity_col],
                    color='green',
                    marker='.',
                    s=1,
                    label=f'{entity} {entity_id}; included')

    else:
        ax1.scatter(entity_df['timestamp'],
                    entity_df[intensity_col],
                    color='blue',
                    marker='.',
                    s=1,
                    label=f'{entity} {entity_id}; hf: {half_life}')

    if included and included in entity_df.columns:
        ax2 = ax1.twinx()
        ax2.scatter(entity_df['timestamp'],
                    entity_df[included],
                    color='green',
                    alpha=0.1,
                    label='Included Events',
                    marker='|',
                    s=5)
        ax2.set_ylabel('Included Events', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(-0.1, 1.1)  # keep it binary-looking

    ax1.set_title(f'Arrival Intensity Over Time for Entity {n}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Arrival Intensity')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confidence_interval(df: pd.DataFrame,
                             entity: str,
                             profile: str,
                             half_life: str,
                             threshold: int,
                             output_path: str,
                             n: int = 1):
    """Plots the confidence interval for reconstructered features

    Args:
      df: Pandas dataframe. Must have an `event_time_unix` column with
        timestamps in miliseconds, an `entity` columns, an `included column`
        and an `intensity_half_life_{half_life}` column.
      profile: The feature to be studied, e.g, sum, count...
      entity: Entity for which to plot, e.g, merchant_id, card_id, device_ip...
      half_life: The half_life for which to plot the intensity function.
      threshold: Filtering threshold for which to plot the reconstruction.
      included: Column name that decides if an event was filtered out or not.
      output_path: Path to where the plot should be saved.
      n: Rank of the entity. n=1 we choose the entity with highest volume.
    """
    df['timestamp'] = pd.to_datetime(df['event_time_unix'], unit='ms')

    top_entity = [df[entity].value_counts().nlargest(n).index.tolist()[-1]]
    entity_df = df[df[entity].isin(top_entity)]

    plt.figure(figsize=(12, 6))

    plt.scatter(entity_df['timestamp'],
                entity_df[f'{profile}_amount_usd'],
                label=f'{entity} {profile}',
                marker='.',
                s=1)

    z = norm.ppf(0.975)
    estimator = entity_df[
        f'{profile}_amount_usd_half_life_{half_life}_threshold_{threshold}']
    std = np.sqrt(entity_df[
        f'var_{profile}_amount_usd_half_life_{half_life}_threshold_{threshold}']
                  ) * z
    lower = estimator - std
    upper = estimator + std

    plt.vlines(x=entity_df['timestamp'],
               ymin=lower,
               ymax=upper,
               color='r',
               alpha=0.1,
               linewidth=.1)

    plt.title(
        f'True profile and reconstructions confidence interval: {profile}')
    plt.xlabel('Time')
    plt.ylabel(f'{profile}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reconstruction(df: pd.DataFrame,
                        entity: str,
                        profile: str,
                        half_life: str,
                        threshold: int,
                        output_path: str,
                        n: int = 1):
    """Plots the confidence interval for reconstructered features

    Args:
      df: Pandas dataframe. Must have an `event_time_unix` column with
        timestamps in miliseconds, an `entity` columns, an `included column`
        and an `intensity_half_life_{half_life}` column.
      profile: The feature to be studied, e.g, sum, count...
      threshold: Filtering threshold for which to plot the reconstruction.
      entity: Entity for which to plot, e.g, merchant_id, card_id, device_ip...
      half_life: The half_life for which to plot the intensity function.
      included: Column name that decides if an event was filtered out or not.
      output_path: Path to where the plot should be saved.
      n: Rank of the entity. n=1 we choose the entity with highest volume.
    """
    df['timestamp'] = pd.to_datetime(df['event_time_unix'], unit='ms')

    top_entity = [df[entity].value_counts().nlargest(n).index.tolist()[-1]]
    entity_df = df[df[entity].isin(top_entity)]

    plt.figure(figsize=(12, 6))

    plt.scatter(entity_df['timestamp'],
                entity_df[f'{profile}_amount_usd'],
                label=f'{entity} {profile}',
                marker='.',
                s=1)

    plt.scatter(
        entity_df['timestamp'],
        entity_df[
            f'{profile}_amount_usd_half_life_{half_life}_threshold_{threshold}'],
        color='r',
        marker='.',
        s=1,
        label='reconstruction')

    plt.title(f'True profile vs estimated: {profile}')
    plt.xlabel('Time')
    plt.ylabel(f'{profile}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
