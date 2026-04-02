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
#define vi vector<int>
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
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1);
    ff(i, 1, n) cin >> a[i];
    vi pre(32, 0);
    ff(i, 1, n)
    {
        for (int j = 0; j < 32; j++)
        {
            if (a[i] & (1LL << j))
                pre[j]++;
        }
    }
    int ans = -1;
    ff(i, 1, n)
    {
        int now = 0;
        for (int j = 0; j < 32; j++)
        {
            if (a[i] & (1LL << j))
                now += (n - pre[j]) * (1LL << j);
            else
                now += (pre[j]) * (1LL << j);
        }
        ans = max(ans, now);
    }
    cout << ans << endl;
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
        // cout << "Case #" << _T + 1 << ": \n";
        Murasame();
    }
    return 0;
}