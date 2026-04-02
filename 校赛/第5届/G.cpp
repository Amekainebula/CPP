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
struct node
{
    int a, b, c, cost, wa;
} prob[15];
void solve()
{
    int n, t, k;
    cin >> n >> t >> k;
    for (int i = 1; i <= n; i++)
    {
        cin >> prob[i].a >> prob[i].b >> prob[i].c >> prob[i].cost >> prob[i].wa;
        prob[i].a -= prob[i].wa * k;
    }
    vector<int> de;
    for (int i = 1; i <= n; i++)
        de.pb(i);
    int ans = 0;
    do
    {
        int tp = 0, tt = t;
        for (int i : de)
        {
            if (tt >= prob[i].cost)
            {
                tt -= prob[i].cost;
                tp += max(prob[i].c, prob[i].a - (t - tt) * prob[i].b);
            }
        }
        ans = max(ans, tp);
    } while (next_permutation(all(de)));
    cout << ans << endl;
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