#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
    double st;
    int k, m;
    cin >> st >> k >> m;
    vector<int> q1(k + 1), q2(k + 1);
    int cnt1 = 0, cnt2 = 0;
    for (int i = 1; i <= k; i++)
    {
        int op;
        double x;
        cin >> op >> x;
        if (op == 0)
            q1[++cnt1] = x;
        else
            q2[++cnt2] = x;
    }
    sort(q1.begin() + 1, q1.begin() + cnt1 + 1);
    sort(q2.begin() + 1, q2.begin() + cnt2 + 1, greater<int>());
    vc<double> arr1(k + 1), arr2(k + 1);
    arr1[0] = 1;
    for (int i = 1; i <= cnt1; i++)
        arr1[i] = arr1[i - 1] * (q1[i] / 10.0);
    ff(i, cnt1 + 1, m)
        arr1[i] = arr1[cnt1] * 1.0;
    for (int i = 1; i <= cnt2; i++)
        arr2[i] = arr2[i - 1] + q2[i]*1.0;
    ff(i, cnt2 + 1, m)
        arr2[i] = arr2[cnt2] * 1.0;
    double ans = 0;
    // for (int i = 0; i <= m; i++)
    //     cout << arr1[i] << " ";
    // cout << endl;
    // for (int i = 0; i <= m; i++)
    //     cout << arr2[i] << " ";
    // cout << endl;
    for (int i = 0; i <= m; i++)
    {
        ans = max(ans, (1 - arr1[i]) * st + arr2[m - i]);
    }
    cout << fixed << setprecision(2) << max(0.0, st - ans) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}