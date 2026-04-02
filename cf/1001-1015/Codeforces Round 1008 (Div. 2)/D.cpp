#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n;
    cin >> n;
    vector<pii> a1(n + 1), a2(n + 1);
    ff(i, 1, n)
    {
        char c1, c2;
        int x1, x2;
        cin >> c1 >> x1 >> c2 >> x2;
        if (c1 == '+')
            a1[i] = {0, x1};
        else
            a1[i] = {1, x1};
        if (c2 == '+')
            a2[i] = {0, x2};
        else
            a2[i] = {1, x2};
    }
    vector<int> path(n + 2, -1);
    ffg(i, n, 1)
    {
        if (a1[i].fi || a2[i].fi)
        {
            if (a1[i].fi == a2[i].fi)
                if (a1[i].se == a2[i].se)
                    path[i] = path[i + 1];
                else
                    path[i] = a1[i].se > a2[i].se ? 1 : 2;
            else
                path[i] = a1[i].fi ? 1 : 2;
        }
        else
            path[i] = path[i + 1];
    }
    int now[3] = {0, 1, 1};
    int temp = 0;
    temp += a1[1].fi ? now[1] * (a1[1].se - 1) : a1[1].se;
    temp += a2[1].fi ? now[2] * (a2[1].se - 1) : a2[1].se;
    ff(i, 2, n)
    {
        if (path[i] == -1)
            now[1] += temp;
        else
            now[path[i]] += temp;
        temp = 0;
        temp += a1[i].fi ? now[1] * (a1[i].se - 1) : a1[i].se;
        temp += a2[i].fi ? now[2] * (a2[i].se - 1) : a2[i].se;
    }
    cout << now[1] + now[2] + temp << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}