#include <bits/stdc++.h>
// Finish Time: 2025/4/20 11:01:23 AC
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
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
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
    string s;
    cin >> s;
    vi ans(n + 1, 0);
    int now = 0;
    ffg(i, n - 2, 0)
    {
        if (s[i] == '<')
            ans[i + 2] = ++now;
    }
    ans[1] = ++now;
    ff(i, 1, n)
    {
        if (ans[i] != 0)
            cout << ans[i] << " ";
        else
            cout << ++now << " ";
    }
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