from pkg_resources import get_distribution

if get_distribution('ray[rllib]').version == '0.8.3':
    print('0.8.3')
elif get_distribution('ray[rllib]').version == '2.0.0':
    print(get_distribution('ray[rllib]').version)
else:
    print('ray[rllib] not supprted')
