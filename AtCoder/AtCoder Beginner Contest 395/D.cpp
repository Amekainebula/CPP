#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
void solve()
{
    int n, q;
    cin >> n >> q;
    vector<int> bird_in_nest_i(n + 1), nest_i(n + 1), i_nest(n + 1);
    for (int i = 1; i <= n; i++)
        bird_in_nest_i[i] = nest_i[i] = i_nest[i] = i;
    while (q--)
    {
        int op;
        cin >> op;
        if (op == 1)
        {
            int x, y;
            cin >> x >> y;
            bird_in_nest_i[x] = nest_i[y];
        }
        else if (op == 2)
        {
            int x, y;
            cin >> x >> y;
            swap(nest_i[x], nest_i[y]);
            swap(i_nest[nest_i[x]], i_nest[nest_i[y]]);
        }
        else
        {
            int x;
            cin >> x;
            cout << i_nest[bird_in_nest_i[x]] << endl;
        }
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}