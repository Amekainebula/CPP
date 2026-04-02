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
    int l, r, id;
};
void solve()
{
    int n;
    cin >> n;
    vector<node> v(n);
    for (int i = 0; i < n; i++)
    {
        cin >> v[i].l >> v[i].r;
        v[i].id = i;
    }
    auto cmp = [](node a, node b)
    {
        if (a.l != b.l)
            return a.l < b.l;
        return a.r < b.r;
    };
    sort(all(v), cmp);
    vector<int> ans;
    int now1 = -1, now2 = -1;
    for (int i = 0; i < n; i++)
    {
        if (v[i].l > now1)
        {
            now1 = v[i].r;
            ans.pb(v[i].id);
        }
        else if (v[i].l > now2)
            now2 = v[i].r;
        else
        {
            cout << "-1" << endl;
            return;
        }
    }
    cout << sz(ans) << endl;
    for (int i = 0; i < sz(ans); i++)
    {
        cout << ans[i] + 1 << " ";
    }
    cout << endl;
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