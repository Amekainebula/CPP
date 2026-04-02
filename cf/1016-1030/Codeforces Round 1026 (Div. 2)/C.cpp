#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
void Murasame()
{
    int n;
    cin >> n;
    stack<int> has;
    vi ans(n);
    vc<pii> d(n);
    int now = 0;
    ff(i, 0, n - 1) cin >> ans[i];
    ff(i, 0, n - 1) cin >> d[i].fi >> d[i].se;
    ff(i, 0, n - 1)
    {
        if (ans[i] == -1)
            has.push(i);
        else
            now += ans[i];
        while (now < d[i].fi)
        {
            if (has.empty())
            {
                cout << -1 << endl;
                return;
            }
            now++;
            ans[has.top()] = 1;
            has.pop();
        }
        while (now + has.size() > d[i].se)
        {
            if (has.empty())
            {
                cout << -1 << endl;
                return;
            }
            ans[has.top()] = 0;
            has.pop();
        }
    }
    for (int i : ans)
        cout << max(i, 0LL) << ' ';
    cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}