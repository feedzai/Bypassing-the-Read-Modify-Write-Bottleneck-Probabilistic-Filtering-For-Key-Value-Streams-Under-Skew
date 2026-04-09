class WindowedCount:
    unfiltered = 'count()'
    filtered = 'sumIf(1/{in_prob}, {included} = 1)'
    variance = 'sumIf((1 - {in_prob})/(power({in_prob}, 2)), {included} = 1)'

    def __str__(self):
        return 'windowed_count'


class TimeDecayedCount:

    def __init__(self, half_life):
        self.half_life = half_life
        self.unfiltered = f'exponentialTimeDecayedCount({half_life})' + '({time})'
        self.filtered = f'exponentialTimeDecayedSum({half_life})' + '(1/{in_prob} * {included}, {time})'
        self.variance = f'exponentialTimeDecayedSum({half_life}/2)'
        self.variance += '({included} * (1 - {in_prob})/(power({in_prob}, 2)), {time})'

    def __str__(self):
        return f'time_decayed_count_{self.half_life}'


class TimeDecayedSum:

    def __init__(self, half_life):
        self.half_life = half_life
        self.unfiltered = f'exponentialTimeDecayedSum({half_life})' + '(coalesce({quantity}, 0), {time})'
        self.filtered = f'exponentialTimeDecayedSum({half_life})' + '(coalesce({quantity}, 0)/{in_prob} * {included}, {time})'
        self.variance = f'exponentialTimeDecayedSum({half_life}/2)'
        self.variance += '({included} * power(coalesce({quantity}, 0), 2) * (1 - {in_prob})/(power({in_prob}, 2)), {time})'

    def __str__(self):
        return f'time_decayed_sum_{self.half_life}'


class TimeDecayedAverage:

    def __init__(self, half_life):
        self.half_life = half_life
        self.unfiltered = f'exponentialTimeDecayedAvg({half_life})' + '(coalesce({quantity}, 0), {time})'

        denominator = f'exponentialTimeDecayedSum({half_life})' + '(1/{in_prob} * {included}, {time})'
        self.filtered = (f'(1/if(abs({denominator} over' +
                         ' {{window}}) < 1e-12, 1, ' +
                         f' {denominator} over ' + '{{window}})) * ')
        self.filtered += f'exponentialTimeDecayedSum({half_life})'
        self.filtered += '(coalesce({quantity}, 0)/{in_prob} * {included}, {time})'
        self.variance = None

    def __str__(self):
        return f'time_decayed_average_{self.half_life}'


class TimeDecayedLogSum:

    def __init__(self, half_life):
        self.half_life = half_life

        log_expr = 'log10(greatest(coalesce({quantity}, 0), .1))'

        self.unfiltered = (f'exponentialTimeDecayedSum({half_life})' +
                           f'({log_expr}, {{time}})')

        self.filtered = (
            f'exponentialTimeDecayedSum({half_life})' +
            f'(({log_expr})/{{in_prob}} * {{included}}, {{time}})')

        self.variance = None
        # self.variance = f'exponentialTimeDecayedSum({half_life}/2)'
        # self.variance += (
        #     f'({{included}} * power({log_expr}, 2) * (1 - {{in_prob}})/(power({{in_prob}}, 2)), {{time}})'
        # )

    def __str__(self):
        return f'time_decayed_log10_sum_{self.half_life}'


class TimeDecayedLogAverage:

    def __init__(self, half_life):
        self.half_life = half_life

        log_expr = 'log10(greatest(coalesce({quantity}, 0), .1))'

        self.unfiltered = (f'exponentialTimeDecayedAvg({half_life})' +
                           f'({log_expr}, {{time}})')

        denominator = f'exponentialTimeDecayedSum({half_life})(1/{{in_prob}} * {{included}}, {{time}})'
        self.filtered = f'1/(if({denominator} over' + ' {{window}} = 0, 1, ' + f'{denominator} over ' + '{{window}})) * '
        self.filtered += (
            f'exponentialTimeDecayedSum({half_life})' +
            f'(({log_expr})/{{in_prob}} * {{included}}, {{time}})')

        self.variance = None

    def __str__(self):
        return f'time_decayed_log10_avg_{self.half_life}'
