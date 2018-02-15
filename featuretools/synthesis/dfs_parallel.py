from datetime import datetime

import distributed
import pandas as pd

from .deep_feature_synthesis import DeepFeatureSynthesis

from featuretools.computational_backends import (
    bin_cutoff_times,
    calculate_feature_matrix,
    calc_num_per_chunk,
)
from featuretools.entityset import EntitySet


def dfs_parallel(entities=None,
                 relationships=None,
                 entityset=None,
                 target_entity=None,
                 cutoff_time=None,
                 instance_ids=None,
                 agg_primitives=None,
                 trans_primitives=None,
                 allowed_paths=None,
                 max_depth=None,
                 ignore_entities=None,
                 ignore_variables=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_primitives=None,
                 max_features=None,
                 cutoff_time_in_index=False,
                 save_progress=None,
                 features_only=False,
                 training_window=None,
                 approximate=None,
                 njobs=None,
                 chunk_size=None,
                 cluster=None,
                 verbose=False):
    '''Calculates a feature matrix and features given a dictionary of entities
    and a list of relationships.


    Args:
        entities (dict[str -> tuple(pd.DataFrame, str, str)]): dictionary of
            entities. Entries take the format
            {entity id -> (dataframe, id column, (time_column))}

        relationships (list[(str, str, str, str)]): list of relationships
            between entities. List items are a tuple with the format
            (parent entity id, parent variable, child entity id, child variable)

        entityset (:class:`.EntitySet`): An already initialized entityset. Required if
            entities and relationships are not defined

        target_entity (str): id of entity to predict on

        cutoff_time (pd.DataFrame or Datetime): specifies what time to calculate
            the features for each instance at.  Can either be a DataFrame with
            'instance_id' and 'time' columns, DataFrame with the name of the
            index variable in the target entity and a time column, a list of values, or a single
            value to calculate for all instances. If the dataframe has more than two columns, any additional
            columns will be added to the resulting feature matrix.

        instance_ids (list): list of instances to calculate features on. Only
            used if cutoff_time is a single datetime.

        agg_primitives (list[:class:`AggregationPrimitive .primitives.AggregationPrimitive`], optional):
            list of Aggregation Feature types to apply.

                Default:[:class:`Sum <.primitives.Sum>`, \
                         :class:`Std <.primitives.Std>`, \
                         :class:`Max <.primitives.Max>`, \
                         :class:`Skew <.primitives.Skew>`, \
                         :class:`Min <.primitives.Min>`, \
                         :class:`Mean <.primitives.Mean>`, \
                         :class:`Count <.primitives.Count>`, \
                         :class:`PercentTrue <.primitives.PercentTrue>`, \
                         :class:`NUniqe <.primitives.NUnique>`, \
                         :class:`Mode <.primitives.Mode>`]

        trans_primitives (list[:class:`TransformPrimitive <.primitives.TransformPrimitive>`], optional):
            list of Transform Feature functions to apply.

                Default:[:class:`Day <.primitives.Day>`, \
                         :class:`Year <.primitives.Year>`, \
                         :class:`Month <.primitives.Month>`, \
                         :class:`Weekday <.primitives.Weekday>`]

        allowed_paths (list[list[str]]): Allowed entity paths to make
            features for

        max_depth (int) : maximum allowed depth of features

        ignore_entities (list[str], optional): List of entities to
            blacklist when creating features

        ignore_variables (dict[str -> str], optional): List of specific
            variables within each entity to blacklist when creating features

        seed_features (list[:class:`.PrimitiveBase`]): List of manually defined
            features to use.

        drop_contains (list[str], optional): drop features
            that contains these strings in name

        drop_exact (list[str], optional): drop features that
            exactly match these strings in name

        where_primitives (list[:class:`.primitives.AggregationPrimitive`], optional):
            list of Aggregation Feature types to apply with where clauses.

        max_features (int, optional) : Cap the number of generated features to
                this number. If -1, no limit.

        features_only (boolean, optional): if True, returns the list of
            features without calculating the feature matrix.

        cutoff_time_in_index (bool): If True, return a DataFrame with a MultiIndex
            where the second index is the cutoff time (first is instance id).
            DataFrame will be sorted by (time, instance_id).

        training_window (dict[str-> :class:`Timedelta`] or :class:`Timedelta`, optional):
            Window or windows defining how much older than the cutoff time data
            can be to be included when calculating the feature.  To specify
            which entities to apply windows to, use a dictionary mapping entity
            id -> Timedelta. If None, all older data is used.

        approximate (Timedelta): bucket size to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        save_progress (Optional(str)): path to save intermediate computational results


    Examples:
        .. code-block:: python

            from featuretools.primitives import Mean
            # cutoff times per instance
            entities = {
                "sessions" : (session_df, "id"),
                "transactions" : (transactions_df, "id", "transaction_time")
            }
            relationships = [("sessions", "id", "transactions", "session_id")]
            feature_matrix, features = dfs(entities=entities,
                                           relationships=relationships,
                                           target_entity="transactions",
                                           cutoff_time=cutoff_times)
            feature_matrix

            features = dfs(entities=entities,
                           relationships=relationships,
                           target_entity="transactions",
                           features_only=True)
    '''
    if not isinstance(entityset, EntitySet):
        entityset = EntitySet("dfs", entities, relationships)

    # set up cluster / client
    if cluster is None:
        temp_cluster = distributed.LocalCluster(n_workers=njobs)
        client = distributed.Client(temp_cluster)
    else:
        client = distributed.Client(cluster)

    dfs_object = DeepFeatureSynthesis(target_entity_id=target_entity,
                                      entityset=entityset,
                                      agg_primitives=agg_primitives,
                                      trans_primitives=trans_primitives,
                                      max_depth=max_depth,
                                      where_primitives=where_primitives,
                                      allowed_paths=allowed_paths,
                                      drop_exact=drop_exact,
                                      drop_contains=drop_contains,
                                      ignore_entities=ignore_entities,
                                      ignore_variables=ignore_variables,
                                      max_features=max_features,
                                      seed_features=seed_features)
    features = dfs_object.build_features()
    feature_futures = client.scatter({'features': features}, broadcast=True)
    if features_only:
        return features

    # construct cutoff_time if necessary
    if not isinstance(cutoff_time, pd.DataFrame):
        if cutoff_time is None:
            cutoff_time = datetime.now()

        if instance_ids is None:
            index_var = entityset[target_entity].index
            instance_ids = entityset[target_entity].df[index_var].tolist()

        if not isinstance(cutoff_time, list):
            cutoff_time = [cutoff_time] * len(instance_ids)

        map_args = [(id, time) for id, time in zip(instance_ids, cutoff_time)]
        df_args = pd.DataFrame(map_args, columns=['instance_id', 'time'])
        to_calc = df_args.values
        cutoff_time = pd.DataFrame(to_calc, columns=['instance_id', 'time'])
    else:
        cutoff_time = cutoff_time.copy()

    num_per_chunk = calc_num_per_chunk(chunk_size, cutoff_time.shape)
    cutoff_df_time_var = 'time'
    target_time = '_original_time'

    # bin cutoff_time if necessary
    if approximate is not None:
        # If there are approximated aggs, bin times
        binned_cutoff_time = bin_cutoff_times(cutoff_time.copy(), approximate)

        # Think about collisions: what if original time is a feature
        binned_cutoff_time[target_time] = cutoff_time[cutoff_df_time_var]

        chunks = chunk_cutoff_time(binned_cutoff_time, cutoff_df_time_var, num_per_chunk)
        for chunk in chunks:
            chunk[cutoff_df_time_var] = chunk[target_time]
            chunk.drop(columns=[target_time], inplace=True)

    else:
        chunks = chunk_cutoff_time(cutoff_time, cutoff_df_time_var, num_per_chunk)


    feature_matrix = []
    for chunk in chunks:
        delayed_fm = client.submit(calculate_feature_matrix,
                                   features=feature_futures['features'],
                                   cutoff_time=chunk,
                                   training_window=training_window,
                                   approximate=approximate,
                                   cutoff_time_in_index=True,
                                   save_progress=save_progress,
                                   verbose=False)
        feature_matrix.append(delayed_fm)

    feature_matrix = client.submit(pd.concat, objs=feature_matrix).result()
    feature_matrix.sort_index(level='time', kind='mergesort', inplace=True)
    if not cutoff_time_in_index:
        feature_matrix.reset_index(level='time', drop=True, inplace=True)

    if cluster is None:
        temp_cluster.close()
    client.close()
    return feature_matrix, features


def chunk_cutoff_time(cutoff_time, time_variable, num_per_chunk):
    # groupby time an calculate size of cutoffs
    grouped = cutoff_time.groupby(time_variable, sort=False)

    # split up groups that are too large
    groups = []
    for _, group in grouped:
        indices = group.index.values.tolist()
        if group.shape[0] > num_per_chunk:
            for i in range(0, group.shape[0], num_per_chunk):
                groups.append(indices[i: i + num_per_chunk])
        else:
            groups.append(indices)

    # sort groups largest to smallest
    groups.sort(key=lambda group: len(group))

    full_chunks = []
    partial_chunks = [[]]

    while len(groups) > 0:
        # get next group
        group = groups.pop()
        found_chunk = False

        for i in range(len(partial_chunks)):
            chunk = partial_chunks[i]
            if len(chunk) + len(group) <= num_per_chunk:
                chunk.extend(group)
                found_chunk = True
                if len(chunk) == num_per_chunk:
                    full_chunks.append(cutoff_time.loc[partial_chunks.pop(i)])
                break
        if not found_chunk:
            partial_chunks.append(group)

    for chunk in partial_chunks:
        full_chunks.append(cutoff_time.loc[chunk])
    return full_chunks
