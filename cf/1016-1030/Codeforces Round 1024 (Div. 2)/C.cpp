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
int nex[4][2] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
void Murasame()
{
    int n;
    cin >> n;
    vc<vi> ans(n + 1, vi(n + 1));
    int nx = (n + 1) / 2, ny = (n + 1) / 2;
    int now = 0;
    ans[nx][ny] = 0;
    int need = 2;
    int cnt = 0;
    while (now <= n * n)
    {
        if (cnt == 0)
        {
            ny++;
            cnt++;
            now++;
            if (now > n * n)
                break;
            ans[nx][ny] = now;
            continue;
        }
        nx += nex[cnt / need][0];
        ny += nex[cnt / need][1];
        cnt++;
        now++;
        if (now > n * n)
            break;
        ans[nx][ny] = now;
        if (cnt == need * 4)
        {
            need += 2;
            cnt = 0;
        }
    }
    ff(i, 1, n) ff(j, 1, n) cout << ans[i][j] << " \n"[j == n];
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